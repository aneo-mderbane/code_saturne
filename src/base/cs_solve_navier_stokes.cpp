/*============================================================================
 * Solve the Navier-Stokes equations.
 *============================================================================*/

/*
  This file is part of code_saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2024 EDF S.A.

  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
  Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/*----------------------------------------------------------------------------*/

#include "cs_defs.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "bft_mem.h"
#include "bft_error.h"
#include "bft_printf.h"

#include "cs_ale.h"
#include "cs_array.h"
#include "cs_assert.h"
#include "cs_atmo.h"
#include "cs_at_data_assim.h"
#include "cs_bad_cells_regularisation.h"
#include "cs_base_accel.h"
#include "cs_balance.h"
#include "cs_blas.h"
#include "cs_boundary_conditions.h"
#include "cs_boundary_conditions_set_coeffs.h"
#include "cs_bw_time_diff.h"
#include "cs_cf_boundary_conditions.h"
#include "cs_cf_compute.h"
#include "cs_convection_diffusion.h"
#include "cs_ctwr.h"
#include "cs_ctwr_source_terms.h"
#include "cs_dispatch.h"
#include "cs_divergence.h"
#include "cs_equation_iterative_solve.h"
#include "cs_equation_param.h"
#include "cs_face_viscosity.h"
#include "cs_fan.h"
#include "cs_field.h"
#include "cs_field_default.h"
#include "cs_field_operator.h"
#include "cs_field_pointer.h"
#include "cs_gui.h"
#include "cs_gradient.h"
#include "cs_head_losses.h"
#include "cs_lagr.h"
#include "cs_lagr_head_losses.h"
#include "cs_mass_source_terms.h"
#include "cs_math.h"
#include "cs_matrix_building.h"
#include "cs_mesh.h"
#include "cs_mesh_quantities.h"
#include "cs_parall.h"
#include "cs_parameters.h"
#include "cs_physical_constants.h"
#include "cs_physical_model.h"
#include "cs_porous_model.h"
#include "cs_pressure_correction.h"
#include "cs_prototypes.h"
#include "cs_rotation.h"
#include "cs_sat_coupling.h"
#include "cs_sles_default.h"
#include "cs_thermal_model.h"
#include "cs_turbulence_ke.h"
//#include "cs_time_step.h"
#include "cs_turbomachinery.h"
#include "cs_turbulence_model.h"
#include "cs_velocity_pressure.h"
#include "cs_vof.h"
#include "cs_volume_mass_injection.h"
#include "cs_wall_condensation.h"
#include "cs_wall_functions.h"

/*----------------------------------------------------------------------------
 * Header for the current file
 *----------------------------------------------------------------------------*/

#include "cs_solve_navier_stokes.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Additional doxygen documentation
 *============================================================================*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*============================================================================
 * Type and macro definitions
 *============================================================================*/

/*============================================================================
 * Static global variables
 *============================================================================*/

/*============================================================================
 * Global variables
 *============================================================================*/

extern cs_real_t *cs_glob_ckupdc;

/*============================================================================
 * Prototypes for Fortran functions and variables.
 *============================================================================*/

/*============================================================================
 * Prototypes for functions intended for use only by Fortran wrappers.
 * (descriptions follow, with function bodies).
 *============================================================================*/

/*============================================================================
 * Fortran function prototypes for subroutines from field.f90.
 *============================================================================*/

void
cs_f_navier_stokes_total_pressure(void);

/*============================================================================
 * Prototypes for functions intended for use only by Fortran wrappers.
 * (descriptions follow, with function bodies).
 *============================================================================*/

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update the convective mass flux before the Navier Stokes equations
 *        (prediction and correction steps) for vp_param->iphydr == 2.
 *
 * This function computes a potential \f$ \varia \f$ solving the equation:
 * \f[
 * D \left( \Delta t, \varia \right) = \divs \left( \rho \vect{u}^n\right)
 *                                   - \Gamma^n
 *                                   + \dfrac{\rho^n - \rho^{n-1}}{\Delta t}
 * \f]
 * This potential is then used to update the mass flux as follows:
 * \f[
 *  \dot{m}^{n+\frac{1}{2}}_\ij = \dot{m}^{n}_\ij
 *                               - \Delta t \grad_\fij \varia \cdot \vect{S}_\ij
 * \f]
 *
 * \param[in]  m   pointer to associated mesh structure
 * \param[in]  mq  pointer to associated mesh quantities structure
 * \param[in]  dt  time step (per cell)
 */
/*----------------------------------------------------------------------------*/

static void
_cs_mass_flux_prediction(const cs_mesh_t       *m,
                         cs_mesh_quantities_t  *mq,
                         cs_real_t              dt[])
{
  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_i_faces = m->n_i_faces;
  const cs_lnum_t n_b_faces = m->n_b_faces;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;

  const cs_real_t *volume = mq->cell_f_vol;

  int idtvar = cs_glob_time_step_options->idtvar;

  cs_dispatch_context ctx;

  const char name[] = "potential";

  /* Physical quantities */
  const cs_real_t *crom = CS_F_(rho)->val;
  const cs_real_t *croma = CS_F_(rho)->val_pre;

  cs_field_bc_coeffs_t bc_coeffs_pot;
  cs_field_bc_coeffs_init(&bc_coeffs_pot);
  CS_MALLOC_HD(bc_coeffs_pot.a, n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs_pot.b, n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs_pot.af, n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs_pot.bf, n_b_faces, cs_real_t, cs_alloc_mode);

  cs_real_t *clapot = bc_coeffs_pot.a;
  cs_real_t *clbpot = bc_coeffs_pot.b;
  cs_real_t *cfapot = bc_coeffs_pot.af;
  cs_real_t *cfbpot = bc_coeffs_pot.bf;

  /* Mass fluxes */
  const int kimasf = cs_field_key_id("inner_mass_flux_id");
  const int kbmasf = cs_field_key_id("boundary_mass_flux_id");
  cs_real_t *imasfl
    = cs_field_by_id(cs_field_get_key_int(CS_F_(vel), kimasf))->val;
  cs_real_t *bmasfl
    = cs_field_by_id(cs_field_get_key_int(CS_F_(vel), kbmasf))->val;

  /* Boundary conditions on the potential (homogenous Neumann) */

  ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {

    /* Homogeneous Neumann BCs */

    // Gradient BCs
    clapot[f_id] = 0.;
    clbpot[f_id] = 1.;

    // Flux BCs
    cfapot[f_id] = 0.;
    cfbpot[f_id] = 0.;
  });

  cs_real_t *divu;
  CS_MALLOC_HD(divu, n_cells_ext, cs_real_t, cs_alloc_mode);

  /* Right Hand side
     --------------- */

  /* Initial mass divergence */
  cs_divergence(m, 1, imasfl, bmasfl, divu);

  /* Mass source terms */

  cs_lnum_t  ncesmp;
  const cs_lnum_t  *icetsm;
  cs_real_t *smacel_p;

  cs_volume_mass_injection_get_arrays(CS_F_(p),
                                      &ncesmp,
                                      &icetsm,
                                      nullptr,
                                      &smacel_p,
                                      nullptr);

  if (ncesmp > 0) {
    ctx.parallel_for(ncesmp, [=] CS_F_HOST_DEVICE (cs_lnum_t cidx) {
      const cs_lnum_t cell_id = icetsm[cidx];
      /* FIXME It should be scmacel at time n-1 */
      divu[cell_id] -= volume[cell_id] * smacel_p[cidx];
    });
  }

  /* Source term associated to the mass aggregation */

  cs_real_t *rhs;
  CS_MALLOC_HD(rhs, n_cells_ext, cs_real_t, cs_alloc_mode);

  ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t cell_id) {
    cs_real_t drom = crom[cell_id] - croma[cell_id];
    divu[cell_id] += drom * volume[cell_id] / dt[cell_id];
    /* The initial Right Hand Side is - div(u) */
    rhs[cell_id] = - divu[cell_id];
  });

  ctx.wait(); // needed to compute the CPU norm

  /* Residual of the system if needed */

  const cs_real_t rnorm = sqrt(cs_gdot(n_cells, rhs, rhs));

  /* Build the linear system to solve
     -------------------------------- */

  /* Unsteady term */

  cs_real_t *pot;
  CS_MALLOC_HD(pot, n_cells_ext, cs_real_t, cs_alloc_mode);
  cs_arrays_set_value<cs_real_t, 1>(n_cells, 0., pot);

  /* Face diffusibility scalar */

  cs_real_t *i_visc, *b_visc;
  CS_MALLOC_HD(i_visc, n_i_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(b_visc, n_b_faces, cs_real_t, cs_alloc_mode);

  const cs_equation_param_t *eqp
    = cs_field_get_equation_param_const(CS_F_(p));

  if (eqp->idiff > 0) {
    cs_face_viscosity(m,
                      mq,
                      eqp->imvisf,
                      dt,
                      i_visc,
                      b_visc);
  }
  else {
    cs_arrays_set_value<cs_real_t, 1>(n_i_faces, 0., i_visc);
    cs_arrays_set_value<cs_real_t, 1>(n_b_faces, 0., b_visc);
  }

  cs_real_t *dam, *xam;
  CS_MALLOC_HD(dam, n_cells_ext, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(xam, n_i_faces, cs_real_t, cs_alloc_mode);

  cs_matrix_wrapper_scalar(eqp->iconv,
                           eqp->idiff,
                           0,   /* strengthen diagonal */
                           1,   /* isym */
                           1.,  /* thetap */
                           0.,  /* imucpp */
                           &bc_coeffs_pot,
                           pot,
                           imasfl,
                           bmasfl,
                           i_visc,
                           b_visc,
                           nullptr,
                           dam,
                           xam);

  /* Solving (Loop over the non-orthogonalities)
     ------------------------------------------- */

  cs_real_t *pota, *dpot;
  CS_MALLOC_HD(pota, n_cells_ext, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(dpot, n_cells_ext, cs_real_t, cs_alloc_mode);

  /* pot     is the potential
   * dpot    is the increment of the potential between sweeps
   * divu    is the inital divergence of the mass flux */

  cs_arrays_set_value<cs_real_t, 1>(n_cells, 0., pot, pota);

  /* (Test to modify if needed: must be sctricly greater than
   * the test in the conjugate gradient) */

  cs_real_t tcrite = 10.0 * eqp->epsrsm * rnorm;

  /* Reconstruction loop (beginning)
     ------------------------------- */

  int isweep = 1;
  cs_real_t residual = rnorm;

  /* logging */
  if (eqp->verbosity > 1)
    cs_log_printf(CS_LOG_DEFAULT,
                  " %s: sweep = %d, RHS norm = %14.6e, relaxp = %f\n",
                  name, isweep, residual, eqp->relaxv);

  while (isweep <= eqp->nswrsm && residual > tcrite) {

    /* Solving on the increment dpot */

    cs_arrays_set_value<cs_real_t, 1>(n_cells, 0., dpot);

    int n_iter = 0;

    cs_sles_solve_native(-1, name,
                         true, /* symmetric */
                         1, 1, /* blocks sizes */
                         dam, xam,
                         eqp->epsilo,
                         rnorm,
                         &n_iter, &residual,
                         rhs, dpot);

    /* Update the increment of potential */

    cs_real_t a;
    if (idtvar >= 0 && isweep <= eqp->nswrsm && residual > tcrite)
      a = eqp->relaxv;
    else
      a = 1.; /* total increment fo last time step */

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t cell_id) {
      pota[cell_id] = pot[cell_id];
      pot[cell_id]  = pota[cell_id] + a*dpot[cell_id];
    });

    isweep += 1;

    /* Update the right hand side if needed:
     * rhs^{k+1} = - div(rho u^n) - D(dt, pot^{k+1}) */

    if (isweep <= eqp->nswrsm) {

      cs_diffusion_potential(-1,
                             m,
                             mq,
                             1,  /* init */
                             0,  /* inc */
                             eqp->imrgra,
                             eqp->nswrgr,
                             eqp->imligr,
                             0,  /* iphydp */
                             eqp->iwgrec,
                             eqp->verbosity,
                             eqp->epsrgr,
                             eqp->climgr,
                             nullptr,
                             pot,
                             &bc_coeffs_pot,
                             i_visc, b_visc,
                             dt,
                             rhs);

      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t cell_id) {
        rhs[cell_id] = - divu[cell_id] - rhs[cell_id];
      });

      ctx.wait(); // needed to compute the CPU residual

      /* Convergence test */
      residual = sqrt(cs_gdot(n_cells, rhs, rhs));

      if (eqp->verbosity > 1) {
        cs_real_t r = (rnorm >= cs_math_epzero) ? residual/rnorm : residual;
        cs_log_printf(CS_LOG_DEFAULT,
                      " %s: sweep = %d, RHS norm = %14.6e, relaxp = %f\n",
                      name, isweep, r, eqp->relaxv);
      }

    }

  } /* End of reconstruction loop */

  if (  isweep > eqp->nswrsm
      && eqp->verbosity > 1) {
    cs_log_printf(CS_LOG_DEFAULT,
                  _("@\n"
                    "@ @@ Warning: %s (mass flux prediction step)\n"
                    "     =======\n"
                    "  Maximum number of iterations (%d) reached\n"),
                  name, eqp->nswrsm);
  }

  /* Update the mass flux
     -------------------- */

  cs_face_diffusion_potential(-1,
                              m,
                              mq,
                              0,  /* init */
                              0,  /* inc */
                              eqp->imrgra,
                              eqp->nswrgr,
                              eqp->imligr,
                              0,  /* iphydp */
                              0,  /* iwgrp */
                              eqp->verbosity,
                              eqp->epsrgr,
                              eqp->climgr,
                              nullptr,
                              pota,
                              &bc_coeffs_pot,
                              i_visc,
                              b_visc,
                              dt,
                              imasfl,
                              bmasfl);

  /* The last increment is not reconstructed to fullfill exactly
     the continuity equation (see theory guide) */

  cs_face_diffusion_potential(-1,
                              m,
                              mq,
                              0,  /* init */
                              0,  /* inc */
                              eqp->imrgra,
                              0,  /* nswrgp */
                              eqp->imligr,
                              0,  /* iphydp */
                              0,  /* iwgrp */
                              eqp->verbosity,
                              eqp->epsrgr,
                              eqp->climgr,
                              nullptr,
                              pota,
                              &bc_coeffs_pot,
                              i_visc,
                              b_visc,
                              dt,
                              imasfl,
                              bmasfl);

  /* Update density (which is coherent with the mass) */

  const cs_fluid_properties_t *fp = cs_glob_fluid_properties;

  if (fp->irovar == 1) {
    const cs_real_t *crom_eos = CS_F_(rho)->val;
    const cs_real_t *brom_eos = CS_F_(rho_b)->val;

    cs_real_t *cpro_rho_mass
      = cs_field_by_name("density_mass")->val;
    cs_real_t *bpro_rho_mass
      = cs_field_by_name("boundary_density_mass")->val;

    cs_array_copy<cs_real_t>(n_cells_ext,
                             (const cs_real_t *)crom_eos,
                             cpro_rho_mass);

    cs_array_copy<cs_real_t>(n_b_faces,
                             (const cs_real_t *)brom_eos,
                             bpro_rho_mass);
  }

  /* Free solver setup
     ----------------- */

  cs_sles_free_native(-1, name);

  CS_FREE_HD(dam);
  CS_FREE_HD(xam);

  CS_FREE_HD(divu);
  CS_FREE_HD(rhs);

  CS_FREE_HD(pot);
  CS_FREE_HD(pota);
  CS_FREE_HD(dpot);

  CS_FREE_HD(clapot);
  CS_FREE_HD(clbpot);
  CS_FREE_HD(cfapot);
  CS_FREE_HD(cfbpot);

  CS_FREE_HD(i_visc);
  CS_FREE_HD(b_visc);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Add the explicit contribution of head loss terms.
 *
 * \param[in]       ncepdp  number of cells with head loss
 * \param[in]       icepdc  index of cells with head loss
 * \param[in]       vela    velocity at the previous time step
 * \param[in]       ckupdc  work array for the head loss
 * \param[in, out]  trav    right hand side
 */
/*----------------------------------------------------------------------------*/

static void
_st_exp_head_loss(cs_lnum_t          ncepdc,
                  const cs_lnum_t    icepdc[],
                  const cs_real_3_t  vela[],
                  const cs_real_6_t  ckupdc[],
                  cs_real_3_t        trav[])
{
  const cs_real_t *crom = CS_F_(rho)->val;
  const cs_real_t *cell_f_vol = cs_glob_mesh_quantities->cell_f_vol;

  cs_dispatch_context ctx;

  ctx.parallel_for(ncepdc, [=] CS_F_HOST_DEVICE (cs_lnum_t hl_id) {

    const cs_lnum_t c_id   = icepdc[hl_id];
    const cs_real_t romvom = -crom[c_id]*cell_f_vol[c_id];
    const cs_real_t cpdc11 = ckupdc[hl_id][0];
    const cs_real_t cpdc22 = ckupdc[hl_id][1];
    const cs_real_t cpdc33 = ckupdc[hl_id][2];
    const cs_real_t cpdc12 = ckupdc[hl_id][3];
    const cs_real_t cpdc23 = ckupdc[hl_id][4];
    const cs_real_t cpdc13 = ckupdc[hl_id][5];
    const cs_real_t vit1   = vela[c_id][0];
    const cs_real_t vit2   = vela[c_id][1];
    const cs_real_t vit3   = vela[c_id][2];

    trav[c_id][0] += romvom*(cpdc11*vit1 + cpdc12*vit2 + cpdc13*vit3);
    trav[c_id][1] += romvom*(cpdc12*vit1 + cpdc22*vit2 + cpdc23*vit3);
    trav[c_id][2] += romvom*(cpdc13*vit1 + cpdc23*vit2 + cpdc33*vit3);
  });

}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update flux mass for turbomachinery.
 *
 * \param[in]      m       pointer to associated mesh structure
 * \param[in]      mq      pointer to associated mesh quantities structure
 * \param[in]      crom    density at cells
 * \param[in]      brom    density at boundary faces
 * \param[in, out] imasfl  interior face mass flux
 * \param[in, out] bmasfl  boundary face mass flux
 */
/*----------------------------------------------------------------------------*/

static void
_turbomachinery_mass_flux(const cs_mesh_t             *m,
                          const cs_mesh_quantities_t  *mq,
                          const cs_real_t              crom[],
                          const cs_real_t              brom[],
                          cs_real_t                    imasfl[],
                          cs_real_t                    bmasfl[])
{
  const cs_lnum_t n_i_faces = m->n_i_faces;
  const cs_lnum_t n_b_faces = m->n_b_faces;

  const cs_lnum_2_t *restrict i_face_cells
    = (const cs_lnum_2_t *restrict)m->i_face_cells;
  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *restrict)m->b_face_cells;

  const cs_real_3_t  *restrict b_face_normal
    = (const cs_real_3_t  *restrict) mq->b_face_normal;
  const cs_real_3_t *restrict i_face_normal
    = (const cs_real_3_t *restrict) mq->i_face_normal;

  const cs_real_3_t *restrict b_face_cog
    = (const cs_real_3_t *restrict)mq->b_face_cog;
  const cs_real_3_t *restrict i_face_cog
    = (const cs_real_3_t *restrict)mq->i_face_cog;

  const int *irotce = cs_turbomachinery_get_cell_rotor_num();

# pragma omp parallel for if (n_i_faces > CS_THR_MIN)
  for (cs_lnum_t face_id = 0; face_id < n_i_faces; face_id++) {

    const cs_lnum_t c_id1 = i_face_cells[face_id][0];
    const cs_lnum_t c_id2 = i_face_cells[face_id][1];
    if ((irotce[c_id1] != 0) || (irotce[c_id2] != 0)) {
      const cs_real_t rhofac = 0.5*(crom[c_id1] + crom[c_id2]);
      cs_real_t vr1[3], vr2[3];
      const cs_rotation_t *r_num1 = cs_glob_rotation + irotce[c_id1];
      const cs_rotation_t *r_num2 = cs_glob_rotation + irotce[c_id2];
      cs_rotation_velocity(r_num1, i_face_cog[face_id], vr1);
      cs_rotation_velocity(r_num2, i_face_cog[face_id], vr2);

      imasfl[face_id] -= 0.5*rhofac*(  i_face_normal[face_id][0]*(vr1[0] + vr2[0])
                                     + i_face_normal[face_id][1]*(vr1[1] + vr2[1])
                                     + i_face_normal[face_id][2]*(vr1[2] + vr2[2]));
    }
  }

# pragma omp parallel if (n_b_faces > CS_THR_MIN)
  for (cs_lnum_t face_id = 0; face_id < n_b_faces; face_id++) {
    cs_real_t vr[3];
    const cs_lnum_t c_id = b_face_cells[face_id];
    if (irotce[c_id] != 0) {
      const cs_real_t rhofac = brom[face_id];
      const cs_rotation_t *r_num = cs_glob_rotation + irotce[c_id];
      cs_rotation_velocity(r_num, b_face_cog[face_id], vr);

      bmasfl[face_id] -= rhofac*(  b_face_normal[face_id][0]*vr[0]
                                 + b_face_normal[face_id][1]*vr[1]
                                 + b_face_normal[face_id][2]*vr[2]);
    }
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Face diffusivity for the velocity
 *
 * \param[in]       m       pointer to associated mesh structure
 * \param[in]       mq      pointer to associated mesh quantities structure
 * \param[in]       eqp_u   pointer to a cs_equation_param_t structure
 * \param[in, out]  viscf   visc*surface/dist at interior faces
 * \param[in, out]  viscb   visc*surface/dist at boundary faces
 * \param[in, out]  viscfi  same as viscf for increments
 * \param[in, out]  viscbi  same as viscb for increments
 * \param[in, out]  viscce  Tensorial diffusion of the velocity
 *                          (in case of tensorial porosity)
 */
/*----------------------------------------------------------------------------*/

static void
_face_diff_vel(const cs_mesh_t             *m,
               const cs_mesh_quantities_t  *mq,
               const cs_equation_param_t   *eqp_u,
               cs_real_t                    viscf[],
               cs_real_t                    viscb[],
               cs_real_t                    viscfi[],
               cs_real_t                    viscbi[],
               cs_real_6_t                  viscce[])
{
  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_i_faces = m->n_i_faces;
  const cs_lnum_t n_b_faces = m->n_b_faces;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;
  cs_dispatch_context ctx;

  if (eqp_u->idiff > 0) {

    const cs_real_t *viscl = CS_F_(mu)->val;
    const cs_real_t *visct = CS_F_(mu_t)->val;
    cs_real_t idifft = eqp_u->idifft;

    cs_real_t *w1;
    CS_MALLOC_HD(w1, n_cells_ext, cs_real_t, cs_alloc_mode);

    if (cs_glob_turb_model->itytur == 3)
      cs_array_copy<cs_real_t>(n_cells, (const cs_real_t *)viscl, w1);
    else {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        w1[c_id] = viscl[c_id] + idifft*visct[c_id];
      });
    }
    /*  Scalar diffusivity (Default) */
    if (eqp_u->idften & CS_ISOTROPIC_DIFFUSION) {

      cs_face_viscosity(m, mq,
                        eqp_u->imvisf,
                        w1,
                        viscf, viscb);

      /* When using Rij-epsilon model with the option irijnu=1, the face
       * viscosity for the Matrix (viscfi and viscbi) is increased */
      if (   cs_glob_turb_model->itytur == 3
          && cs_glob_turb_rans_model->irijnu == 1) {
        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          w1[c_id] = viscl[c_id] + idifft*visct[c_id];
        });

        cs_face_viscosity(m, mq,
                          eqp_u->imvisf,
                          w1,
                          viscfi, viscbi);
      }
    }

    /* Tensorial diffusion of the velocity (in case of tensorial porosity) */
    else if (eqp_u->idften & CS_ANISOTROPIC_LEFT_DIFFUSION) {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          viscce[c_id][ii] = w1[c_id];
        for (cs_lnum_t ii = 3; ii < 6; ii++)
          viscce[c_id][ii] = 0;
      });

      cs_face_anisotropic_viscosity_vector(m, mq,
                                           eqp_u->imvisf,
                                           viscce,
                                           (cs_real_33_t*)viscf,
                                           viscb);

      /* When using Rij-epsilon model with the option irijnu=1, the face
       * viscosity for the Matrix (viscfi and viscbi) is increased */
      if (   cs_glob_turb_model->itytur == 3
          && cs_glob_turb_rans_model->irijnu == 1) {
        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          w1[c_id] = viscl[c_id] + idifft*visct[c_id];

          for (cs_lnum_t ii = 0; ii < 3; ii++)
            viscce[c_id][ii] = w1[c_id];
          for (cs_lnum_t ii = 3; ii < 6; ii++)
            viscce[c_id][ii] = 0;
        });

        cs_face_anisotropic_viscosity_vector(m, mq,
                                             eqp_u->imvisf,
                                             viscce,
                                             (cs_real_33_t*)viscfi,
                                             viscbi);
      }
    }

    CS_FREE_HD(w1);
  }

  /* If no diffusion, viscosity is set to 0. */
  else {

    if (   cs_glob_turb_model->itytur == 3
        && cs_glob_turb_rans_model->irijnu == 1) {

      cs_arrays_set_value<cs_real_t, 1>(n_i_faces, 0., viscf, viscfi);
      cs_arrays_set_value<cs_real_t, 1>(n_b_faces, 0., viscb, viscbi);
    }
    else {
      cs_arrays_set_value<cs_real_t, 1>(n_i_faces, 0., viscf);
      cs_arrays_set_value<cs_real_t, 1>(n_b_faces, 0., viscb);
    }
  }

}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Divergence of tensor Rij
 *        Non linear part of Rij for non-liear Eddy Viscosity Models
 *
 * \param[in]        m         pointer to associated mesh structure
 * \param[in]        crom      density at cells
 * \param[in]        brom      density at boundary faces
 * \param[in, out]   cpro_divr reynolds stress divergence
 * \param[in, out]   c_st_vel  source term of velicity
 * \param[in, out]   forbr     boundary forces
 * \param[in, out]   trava     working array for the
 *                             velocity-pressure coupling
 * \param[in, out]   trav      right hand side for the normalizing
 *                             the residual
 */
/*----------------------------------------------------------------------------*/

static void
_div_rij(const cs_mesh_t     *m,
         const cs_real_t      crom[],
         const cs_real_t      brom[],
         cs_real_3_t         *cpro_divr,
         cs_real_3_t         *c_st_vel,
         cs_real_3_t         *forbr,
         cs_real_3_t          trava[],
         cs_real_3_t          trav[])
{
  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_i_faces = m->n_i_faces;
  const cs_lnum_t n_b_faces = m->n_b_faces;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;

  const cs_mesh_quantities_t *mq = cs_glob_mesh_quantities;
  const cs_velocity_pressure_param_t *vp_param
    = cs_glob_velocity_pressure_param;

  cs_dispatch_context ctx;

  /* Flux computation options */
  const cs_equation_param_t *eqp = nullptr;

  cs_real_3_t *tflmas = nullptr, *tflmab = nullptr;
  CS_MALLOC_HD(tflmas, n_i_faces,cs_real_3_t, cs_alloc_mode);
  CS_MALLOC_HD(tflmab, n_b_faces,cs_real_3_t, cs_alloc_mode);

  /* Reynolds Stress Models */
  if (cs_glob_turb_model->itytur == 3) {

    const cs_field_t *f_rij = CS_F_(rij);
    eqp = cs_field_get_equation_param_const(f_rij);

    cs_field_bc_coeffs_t bc_coeffs_ts_loc;
    cs_field_bc_coeffs_shallow_copy(f_rij->bc_coeffs, &bc_coeffs_ts_loc);

    bc_coeffs_ts_loc.a = f_rij->bc_coeffs->ad;
    bc_coeffs_ts_loc.b = f_rij->bc_coeffs->bd;

    cs_tensor_face_flux(m, mq,
                        -1, 1, 0, 1, 1,
                        eqp->imrgra, eqp->nswrgr,
                        static_cast<cs_gradient_limit_t>(eqp->imligr),
                        eqp->verbosity,
                        eqp->epsrgr, eqp->climgr,
                        crom, brom,
                        (const cs_real_6_t *)f_rij->val,
                        &bc_coeffs_ts_loc,
                        tflmas, tflmab);

    bc_coeffs_ts_loc.a = nullptr;
    bc_coeffs_ts_loc.b = nullptr;
    cs_field_bc_coeffs_free_copy(f_rij->bc_coeffs, &bc_coeffs_ts_loc);
  }

  /* Baglietto et al. quadratic k-epislon model */
  else if (cs_glob_turb_model->iturb == CS_TURB_K_EPSILON_QUAD) {

    cs_real_6_t *rij = nullptr;
    CS_MALLOC_HD(rij, n_cells_ext, cs_real_6_t, cs_alloc_mode);

    cs_field_bc_coeffs_t bc_coeffs_loc;
    cs_field_bc_coeffs_init(&bc_coeffs_loc);
    CS_MALLOC_HD(bc_coeffs_loc.a, 6*n_b_faces, cs_real_t, cs_alloc_mode);
    CS_MALLOC_HD(bc_coeffs_loc.b, 36*n_b_faces, cs_real_t, cs_alloc_mode);
    cs_real_6_t  *coefat = (cs_real_6_t  *)bc_coeffs_loc.a;
    cs_real_66_t *coefbt = (cs_real_66_t *)bc_coeffs_loc.b;

    eqp = cs_field_get_equation_param_const(CS_F_(k));

    /* Compute the non linear part of Rij */
    cs_turbulence_ke_q(-1, rij);

    /* Boundary conditions: homogeneous Neumann */

    cs_arrays_set_value<cs_real_t, 1>(6*n_b_faces, 0., (cs_real_t *)coefat);

    ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
      for (cs_lnum_t jj = 0; jj < 6; jj++) {
        for (cs_lnum_t kk = 0; kk < 6; kk++)
          coefbt[face_id][jj][kk] = 0.;
        coefbt[face_id][jj][jj] = 1.;
      }
    });

    cs_tensor_face_flux(m, mq,
                        -1, 1, 0, 1, 1,
                        eqp->imrgra, eqp->nswrgr,
                        static_cast<cs_gradient_limit_t>(eqp->imligr),
                        eqp->verbosity,
                        eqp->epsrgr, eqp->climgr,
                        crom, brom,
                        rij,
                        &bc_coeffs_loc,
                        tflmas, tflmab);
    CS_FREE_HD(rij);
    CS_FREE_HD(bc_coeffs_loc.a);
    CS_FREE_HD(bc_coeffs_loc.b);

  }

  /* Compute stresses at boundary (part 5/5), if necessary */
  if (forbr != nullptr) {
    ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        forbr[f_id][ii] += tflmab[f_id][ii];
    });
  }

  cs_tensor_divergence(m, 1, tflmas, tflmab, cpro_divr);

  CS_FREE_HD(tflmas);
  CS_FREE_HD(tflmab);

  /* (if iphydr=1 then this term is already taken into account) */

  if (   vp_param->iphydr != 1
      || vp_param->igprij != 1) {

    /* If extrapolation of source terms */
    if (cs_glob_time_scheme->isno2t > 0)
      cs_axpy(n_cells*3, -1, (cs_real_t *)cpro_divr, (cs_real_t *)c_st_vel);

    /* No extrapolation of source terms */
    else {
      /* No inner iteration */
      if (vp_param->nterup == 1)
        cs_axpy(n_cells*3, -1, (cs_real_t *)cpro_divr, (cs_real_t *)trav);

      else
        cs_axpy(n_cells*3, -1, (cs_real_t *)cpro_divr, (cs_real_t *)trava);
    }
  }

}

/*----------------------------------------------------------------------------*/
/*!
 * \brief In the ALE framework, update mass flux by adding mesh velocity.
 *
 * \param[in]      m       pointer to associated mesh structure
 * \param[in]      mq      pointer to associated mesh quantities structure
 * \param[in]      dt      time step at cells
 * \param[in]      crom    density at cells
 * \param[in]      brom    density at boundary faces
 * \param[in, out] imasfl  interior face mass flux
 * \param[in, out] bmasfl  boundary face mass flux
 */
/*----------------------------------------------------------------------------*/

static void
_mesh_velocity_mass_flux(const cs_mesh_t             *m,
                         const cs_mesh_quantities_t  *mq,
                         const cs_real_t              dt[],
                         const cs_real_t              crom[],
                         const cs_real_t              brom[],
                         cs_real_t                    imasfl[],
                         cs_real_t                    bmasfl[])
{
  const cs_lnum_t n_i_faces = m->n_i_faces;
  const cs_lnum_t n_b_faces = m->n_b_faces;

  const cs_lnum_2_t *restrict i_face_cells
    = (const cs_lnum_2_t *restrict)m->i_face_cells;
  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *restrict)m->b_face_cells;

  const cs_lnum_t *i_face_vtx_idx = m->i_face_vtx_idx;
  const cs_lnum_t *i_face_vtx_lst = m->i_face_vtx_lst;
  const cs_lnum_t *b_face_vtx_idx = m->b_face_vtx_idx;
  const cs_lnum_t *b_face_vtx_lst = m->b_face_vtx_lst;

  const cs_real_3_t *vtx_coord = (const cs_real_3_t *)(m->vtx_coord);
  const cs_real_3_t *b_face_normal = (const cs_real_3_t *) mq->b_face_normal;
  const cs_real_3_t *i_face_normal = (const cs_real_3_t *) mq->i_face_normal;

  const cs_real_3_t *mshvel = (const cs_real_3_t *)CS_F_(mesh_u)->val;

  const cs_real_3_t *xyzno0
    = (const cs_real_3_t *)cs_field_by_name("vtx_coord0")->val;

  const cs_real_3_t *disale
    = (const cs_real_3_t *)cs_field_by_name("mesh_displacement")->val;

  cs_dispatch_context ctx;

  if (cs_glob_space_disc->iflxmw > 0) {

    /* One temporary array needed for internal faces,
     * in case some internal vertices are moved directly by the user */

    cs_real_t *intflx = nullptr, *bouflx = nullptr;
    CS_MALLOC_HD(intflx, n_i_faces, cs_real_t, cs_alloc_mode);
    CS_MALLOC_HD(bouflx, n_b_faces, cs_real_t, cs_alloc_mode);

    cs_field_bc_coeffs_t *bc_coeffs_ale = CS_F_(mesh_u)->bc_coeffs;

    const cs_equation_param_t *eqp_mesh
      = cs_field_get_equation_param_const(CS_F_(mesh_u));

    cs_mass_flux(m,
                 mq,
                 CS_F_(mesh_u)->id,
                 1,  /* itypfl */
                 1,  /* iflmb0 */
                 1,  /* init */
                 1,  /* inc */
                 eqp_mesh->imrgra,
                 eqp_mesh->nswrgr,
                 static_cast<cs_gradient_limit_t>(eqp_mesh->imligr),
                 eqp_mesh->verbosity,
                 eqp_mesh->epsrgr,
                 eqp_mesh->climgr,
                 crom, brom,
                 mshvel,
                 bc_coeffs_ale,
                 intflx, bouflx);

    cs_axpy(n_b_faces, -1, bouflx, bmasfl);
    cs_axpy(n_i_faces, -1, intflx, imasfl);

    CS_FREE_HD(intflx);
    CS_FREE_HD(bouflx);
  }

  /* Here we need of the opposite of the mesh velocity. */

  else { /* if (cs_glob_space_disc->iflxmw == 0) */

    /* Compute the mass flux using the nodes displacement */

    ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
      cs_real_t disp_fac[3] = {0, 0, 0};
      const cs_lnum_t s_id = b_face_vtx_idx[face_id];
      const cs_lnum_t e_id = b_face_vtx_idx[face_id+1];
      const cs_lnum_t icpt = e_id - s_id;
      for (cs_lnum_t ii = s_id; ii < e_id; ii++) {
        const cs_lnum_t inod = b_face_vtx_lst[ii];
        for (cs_lnum_t jj = 0; jj < 3; jj++)
          disp_fac[jj] +=   disale[inod][jj]
                          - (vtx_coord[inod][jj] - xyzno0[inod][jj]);
      }
      const cs_lnum_t c_id = b_face_cells[face_id];
      bmasfl[face_id] -= brom[face_id] * (  disp_fac[0]*b_face_normal[face_id][0]
                                          + disp_fac[1]*b_face_normal[face_id][1]
                                          + disp_fac[2]*b_face_normal[face_id][2])
                         / dt[c_id]/icpt;
    });

    ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
      cs_real_t disp_fac[3] = {0, 0, 0};
      const cs_lnum_t s_id = i_face_vtx_idx[face_id];
      const cs_lnum_t e_id = i_face_vtx_idx[face_id+1];
      const cs_lnum_t icpt = e_id - s_id;
      for (cs_lnum_t ii = s_id; ii < e_id; ii++) {
        const cs_lnum_t inod = i_face_vtx_lst[ii];
        for (cs_lnum_t jj = 0; jj < 3; jj++)
          disp_fac[jj] +=   disale[inod][jj]
                          - (vtx_coord[inod][jj] - xyzno0[inod][jj]);
      }

      /* For inner vertices, the mass flux due to the mesh displacement is
       * recomputed from the nodes displacement */
      const cs_lnum_t c_id1 = i_face_cells[face_id][0];
      const cs_lnum_t c_id2 = i_face_cells[face_id][1];
      const cs_real_t dtfac = 0.5*(dt[c_id1] + dt[c_id2]);
      const cs_real_t rhofac = 0.5*(crom[c_id1] + crom[c_id2]);
      imasfl[face_id] -= rhofac * (  disp_fac[0]*i_face_normal[face_id][0]
                                   + disp_fac[1]*i_face_normal[face_id][1]
                                   + disp_fac[2]*i_face_normal[face_id][2])
                         / dtfac / icpt;
    });

  }

  ctx.wait();
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Take external forces partially equilibrated
 *        with the pressure gradient into account
 *
 * \param[in]       m           pointer to associated mesh structure
 * \param[in]       mq          pointer to associated mesh quantities structure
 * \param[in]       fp          pointer to fluid properties structure
 * \param[in]       ncepdc      number of cells in which a pressure drop
 *                              is imposed.
 * \param[in]       icepdc      number of the ncepdc cells in which a pressure
 *                              drop is imposed
 * \param[in]       crom        density at cells
 * \param[in]       croma       density at cells at the previous time
 * \param[in]       cromaa      density at cells at the two previous time
 * \param[in]       gxyz        gravity
 * \param[in]       vela        velocity at the previous time
 * \param[in]       tsexp       explicite source term
 * \param[in]       frcxt       external forces
 * \param[in]       cpro_divr   reynolds stress divergence
 * \param[in]       stf         surface tension force for VoF
 * \param[in]       ckupdc      value of the coefficients of the pressure
 * \param[in]                   drop tensor of the ncepdc cells in which
 * \param[in]                   a pressure drop is imposed.
 * \param[in, out]  dfrcxt      variation of the external forces
 */
/*----------------------------------------------------------------------------*/

static void
_ext_forces(const cs_mesh_t                *m,
            const cs_mesh_quantities_t     *mq,
            const cs_fluid_properties_t    *fp,
            const cs_lnum_t                ncepdc,
            const cs_lnum_t                *icepdc,
            const cs_real_t                crom[],
            const cs_real_t                croma[],
            const cs_real_t                cromaa[],
            const cs_real_3_t              gxyz,
            const cs_real_3_t              vela[],
            const cs_real_3_t              tsexp[],
            const cs_real_3_t              frcxt[],
            const cs_real_3_t              cpro_divr[],
            const cs_real_3_t              stf[],
            const cs_real_6_t              ckupdc[],
            cs_real_3_t                    dfrcxt[])
{
  const cs_lnum_t n_cells = m->n_cells;
  const cs_real_t *cell_f_vol = mq->cell_f_vol;
  /* External forces at previous time step:
   * frcxt was initialized to 0
   * NB: frcxt was used in cs_boundary_conditions_type, and will be updated
   *     at the end of cs_solve_navier_stokes.
   *
   * External force variation between time step n and n+1
   * (used in the correction step) */

  cs_dispatch_context ctx;
  const cs_real_t ro0 = fp->ro0;

  int has_disable_flag = mq->has_disable_flag;
  int *c_disable_flag = mq->c_disable_flag;

  /* Boussinesq approximation */
  if (cs_glob_velocity_pressure_model->idilat == 0) {
    const cs_real_t *cvar_t = cs_thermal_model_field()->val;
    const cs_real_t *cpro_beta = cs_field_by_name("thermal_expansion")->val;

    cs_real_t tref = fp->t0;
    /* for atmospheric flows, variable is potential temperature */
    if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] > CS_ATMO_CONSTANT_DENSITY) {
      const cs_real_t rscp = fp->r_pg_cnst/fp->cp0;
      tref = fp->t0*pow(cs_glob_atmo_constants->ps/fp->p0, rscp);
    }

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      const int ind = has_disable_flag * c_id;
      const int c_act = (1 - (has_disable_flag * c_disable_flag[ind]));
      const cs_real_t drom
        = -crom[c_id]*cpro_beta[c_id]*(cvar_t[c_id] - tref)*c_act;
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        dfrcxt[c_id][ii] = drom*gxyz[ii] - frcxt[c_id][ii]*c_act;
    });
  }

  else {
    int time_order = 1;
    if (   cs_glob_time_scheme->time_order == 2
        && cs_glob_velocity_pressure_param->itpcol == 1)
      time_order = 2;

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      const int ind = has_disable_flag * c_id;
      const int c_act = (1 - (has_disable_flag * c_disable_flag[ind]));
      cs_real_t drom;
      if (time_order == 2)
        drom = (1.5*croma[c_id] - 0.5*cromaa[c_id] - ro0) * c_act;
      else
        drom = (crom[c_id] - ro0) * c_act;

      for (cs_lnum_t ii = 0; ii < 3; ii++)
        dfrcxt[c_id][ii] = drom * gxyz[ii] - frcxt[c_id][ii] * c_act;
    });
  }

  /* Add head losses */
  if (ncepdc > 0) {
    ctx.parallel_for(ncepdc, [=] CS_F_HOST_DEVICE (cs_lnum_t id) {
      const cs_lnum_t c_id = icepdc[id];
      const int ind = has_disable_flag * c_id;
      const int c_act = (1 - (has_disable_flag * c_disable_flag[ind]));
      const cs_real_t vit1   = vela[c_id][0] * c_act;
      const cs_real_t vit2   = vela[c_id][1] * c_act;
      const cs_real_t vit3   = vela[c_id][2] * c_act;
      const cs_real_t cpdc11 = ckupdc[id][0];
      const cs_real_t cpdc22 = ckupdc[id][1];
      const cs_real_t cpdc33 = ckupdc[id][2];
      const cs_real_t cpdc12 = ckupdc[id][3];
      const cs_real_t cpdc23 = ckupdc[id][4];
      const cs_real_t cpdc13 = ckupdc[id][5];

      dfrcxt[c_id][0] -= crom[c_id]*(cpdc11*vit1+cpdc12*vit2+cpdc13*vit3);
      dfrcxt[c_id][1] -= crom[c_id]*(cpdc12*vit1+cpdc22*vit2+cpdc23*vit3);
      dfrcxt[c_id][2] -= crom[c_id]*(cpdc13*vit1+cpdc23*vit2+cpdc33*vit3);
    });
  }

  /* Add Coriolis force */
  cs_turbomachinery_model_t iturbo = cs_turbomachinery_get_model();
  if (   cs_glob_physical_constants->icorio == 1
      || iturbo == CS_TURBOMACHINERY_FROZEN) {
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      const int c_act = cs_mesh_quantities_cell_is_active(mq, c_id);
      const cs_real_t rom = -2*crom[c_id] * c_act;
      cs_rotation_add_coriolis_v(cs_glob_rotation, rom, vela[c_id], dfrcxt[c_id]);
    }

    if (iturbo == CS_TURBOMACHINERY_FROZEN) {
      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        const int *irotce = cs_turbomachinery_get_cell_rotor_num();
        if (irotce[c_id] > 0) {
          const int c_act = cs_mesh_quantities_cell_is_active(mq, c_id);
          const cs_real_t rom = -crom[c_id] * c_act;
          cs_rotation_add_coriolis_v(cs_glob_rotation + irotce[c_id],
                                     rom, vela[c_id], dfrcxt[c_id]);

        }
      }
    }
  }

  /* Add -div( rho R) as external force */
  if (   cs_glob_turb_model->itytur == 3
      && cs_glob_velocity_pressure_param->igprij == 1) {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      cs_real_t dvol = 0;
      const int ind = has_disable_flag * c_id;
      const int c_act = (1 - (has_disable_flag * c_disable_flag[ind]));
      if (c_act == 1)
        dvol = 1.0/cell_f_vol[c_id];
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        dfrcxt[c_id][ii] -= cpro_divr[c_id][ii]*dvol;
    });
  }

  /* Surface tension force for VoF */
  if (stf != nullptr) {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      cs_real_t dvol = 0;
      const int ind = has_disable_flag * c_id;
      const int c_act = (1 - (has_disable_flag * c_disable_flag[ind]));
      /* If it is not a solid cell */
      if (c_act == 1)
        dvol = 1.0/cell_f_vol[c_id];
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        dfrcxt[c_id][ii] += stf[c_id][ii]*dvol;
    });
  }

  /* Use user source terms */
  if (cs_glob_velocity_pressure_param->igpust == 1) {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      cs_real_t dvol = 0;
      const int ind = has_disable_flag * c_id;
      const int c_act = (1 - (has_disable_flag * c_disable_flag[ind]));
      if (c_act == 1)
        dvol = 1.0/cell_f_vol[c_id];
      /* FIXME we should add tsimp*vela to tsexp as for head losses */
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        dfrcxt[c_id][ii] += tsexp[c_id][ii] *dvol;
    });
  }

  ctx.wait(); // needed for the next synchronization

  cs_mesh_sync_var_vect((cs_real_t *)dfrcxt);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update of the fluid velocity field.
 *
 * \param[in]      m       pointer to associated mesh structure
 * \param[in]      mq      pointer to associated mesh quantities structure
 * \param[in]      dt      time step at cells
 * \param[in]      crom    density at cells
 * \param[in]      cromk1  density at cells
 * \param[in, out] imasfl  interior face mass flux
 * \param[in, out] bmasfl  boundary face mass flux
 */
/*----------------------------------------------------------------------------*/

static void
_update_fluid_vel(const cs_mesh_t             *m,
                  const cs_mesh_quantities_t  *mq,
                  const cs_equation_param_t   *eqp_p,
                  const cs_vof_parameters_t   *vof_param,
                  const cs_real_t              dt[],
                  const cs_real_t              crom[],
                  const cs_real_t              cromk1[],
                  cs_real_t                    imasfl[],
                  cs_real_t                    bmasfl[],
                  cs_real_t                    coefa_dp[],
                  cs_real_3_t                  vel[],
                  cs_real_3_t                  dfrcxt[],
                  cs_real_3_t                  frcxt[],
                  cs_real_6_t                  dttens[],
                  const int                    isostd[])
{
  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;
  const cs_lnum_t n_b_faces = m->n_b_faces;

  const cs_lnum_2_t *restrict i_face_cells
    = (const cs_lnum_2_t *restrict)m->i_face_cells;
  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *restrict)m->b_face_cells;

  const cs_real_3_t *restrict i_face_cog
    = (const cs_real_3_t *restrict)mq->i_face_cog;
  const cs_real_3_t *restrict b_face_cog
    = (const cs_real_3_t *restrict)mq->b_face_cog;
  const cs_real_3_t *cell_cen = (const cs_real_3_t *)mq->cell_cen;

  int has_disable_flag = mq->has_disable_flag;
  int *c_disable_flag = mq->c_disable_flag;

  const cs_velocity_pressure_param_t *vp_param = cs_glob_velocity_pressure_param;

  /* Parallel or device dispatch */
  cs_dispatch_context ctx;
  cs_dispatch_sum_type_t i_sum_type = ctx.get_parallel_for_i_faces_sum_type(m);
  cs_dispatch_sum_type_t b_sum_type = ctx.get_parallel_for_b_faces_sum_type(m);

  /* irevmc = 0: Update the velocity with the pressure gradient. */

  if (vp_param->irevmc == 0) {

    /* The predicted velocity is corrected by the cell gradient of the
     * pressure increment. */
    int inc = 0;

    cs_gradient_porosity_balance(inc);

    if (vp_param->iphydr == 1 || vp_param->iifren == 1)
      inc = 1;

    /* Pressure increment gradient */

    cs_real_3_t *cpro_gradp = nullptr, *gradp = nullptr;
    cs_field_t *f_inc = cs_field_by_name_try("algo:gradient_pressure_increment");
    if (f_inc != nullptr)
      cpro_gradp = (cs_real_3_t *)f_inc->val;
    else {
      CS_MALLOC_HD(gradp, n_cells_ext, cs_real_3_t, cs_alloc_mode);
      cpro_gradp = gradp;
    }

    /* Scalar diffusivity */

    cs_real_t *cpro_wgrec_s = nullptr;
    cs_real_6_t *cpro_wgrec_v = nullptr;

    if (vof_param->vof_model != 0) {
      const int kwgrec = cs_field_key_id_try("gradient_weighting_id");
      const int iflwgr = cs_field_get_key_int(CS_F_(p), kwgrec);
      cs_field_t *f_g = cs_field_by_id(iflwgr);
      if (f_g->dim == 1) {
        cpro_wgrec_s = f_g->val;

        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          cpro_wgrec_s[c_id] = dt[c_id] / crom[c_id];
        });
        ctx.wait(); // needed for the next synchronization

        cs_mesh_sync_var_scal(cpro_wgrec_s);
      }
      else if (f_g->dim == 6) {
        cpro_wgrec_v = (cs_real_6_t *)f_g->val;

        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          for (cs_lnum_t ii = 0; ii < 6; ii++)
            cpro_wgrec_v[c_id][ii] = dttens[c_id][ii] / crom[c_id];
        });
        ctx.wait(); // needed for the next synchronization

        cs_mesh_sync_var_sym_tens(cpro_wgrec_v);
      }
    }

    if (cs_glob_velocity_pressure_model->iprcdo == 0) {
      const cs_field_t *f_ddp = cs_field_by_name("pressure_increment");
      cs_field_gradient_potential(f_ddp,
                                  false,
                                  inc,
                                  vp_param->iphydr,
                                  dfrcxt,
                                  cpro_gradp);
    }

    /*  Update the velocity field */

    const cs_real_t thetap = eqp_p->theta;

    /* Specific handling of hydrostatic pressure */

    if (vp_param->iphydr == 1) {

      /* Scalar diffusion for the pressure */
      if (eqp_p->idften & CS_ISOTROPIC_DIFFUSION) {
        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          const cs_real_t dtsrom = thetap*dt[c_id] / crom[c_id];
          const cs_real_t rhok1drhok = cromk1[c_id] / crom[c_id];
          for (cs_lnum_t isou = 0; isou < 3; isou++)
            vel[c_id][isou] =   vel[c_id][isou] * rhok1drhok
                              + dtsrom*(  dfrcxt[c_id][isou]
                                        - cpro_gradp[c_id][isou]);
        });
      }

      /* Tensorial diffusion for the pressure */
      else if (eqp_p->idften & CS_ANISOTROPIC_DIFFUSION) {

        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          const cs_real_t unsrom = thetap / crom[c_id];
          const cs_real_t rhok1drhok = cromk1[c_id] / crom[c_id];

          vel[c_id][0]
            =   vel[c_id][0] * rhok1drhok
              + unsrom*(  dttens[c_id][0]*(dfrcxt[c_id][0]-cpro_gradp[c_id][0])
                        + dttens[c_id][3]*(dfrcxt[c_id][1]-cpro_gradp[c_id][1])
                        + dttens[c_id][5]*(dfrcxt[c_id][2]-cpro_gradp[c_id][2]));
          vel[c_id][1]
            =   vel[c_id][1] * rhok1drhok
              + unsrom*(  dttens[c_id][3]*(dfrcxt[c_id][0]-cpro_gradp[c_id][0])
                        + dttens[c_id][1]*(dfrcxt[c_id][1]-cpro_gradp[c_id][1])
                        + dttens[c_id][4]*(dfrcxt[c_id][2]-cpro_gradp[c_id][2]));

          vel[c_id][2]
            =   vel[c_id][2] * rhok1drhok
              + unsrom*(  dttens[c_id][5]*(dfrcxt[c_id][0]-cpro_gradp[c_id][0])
                        + dttens[c_id][4]*(dfrcxt[c_id][1]-cpro_gradp[c_id][1])
                        + dttens[c_id][2]*(dfrcxt[c_id][2]-cpro_gradp[c_id][2]));
        });

      }

      /* Update of the Dirichlet boundary conditions on the
       * pressure for the outlet */

      const int *iautom = nullptr;
      if (   cs_glob_atmo_option->open_bcs_treatment > 0
          && cs_glob_atmo_option->meteo_profile > 0) {
        iautom = cs_glob_bc_pm_info->iautom;
      }

      cs_real_t *coefa_p = CS_F_(p)->bc_coeffs->a;

      ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
        /*  automatic inlet/outlet face for atmospheric flow */
        int iautof = 0;
        if (iautom != nullptr)
          iautof = iautom[face_id];

        if (isostd[face_id] == 1 || iautof > 0)
          coefa_p[face_id] += coefa_dp[face_id];
      });

    }

    /* Standard handling of hydrostatic pressure */

    else {  /* if (vp_param->iphydr == 0) */

      /* Scalar diffusion for the pressure */
      if (eqp_p->idften & CS_ISOTROPIC_DIFFUSION) {
        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          const cs_real_t dtsrom = thetap*dt[c_id] / crom[c_id];
          const cs_real_t rhok1drhok = cromk1[c_id] / crom[c_id];
          for (cs_lnum_t isou = 0; isou < 3; isou++) {
            vel[c_id][isou] =   vel[c_id][isou] * rhok1drhok
                              - dtsrom * cpro_gradp[c_id][isou];

          }
        });
      }

      /* Tensorial diffusion for the pressure */
      else if (eqp_p->idften & CS_ANISOTROPIC_DIFFUSION) {
        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          const cs_real_t unsrom = thetap / crom[c_id];
          const cs_real_t rhok1drhok = cromk1[c_id] / crom[c_id];

          vel[c_id][0] =   vel[c_id][0] * rhok1drhok
                         - unsrom*(  dttens[c_id][0]*(cpro_gradp[c_id][0])
                                   + dttens[c_id][3]*(cpro_gradp[c_id][1])
                                   + dttens[c_id][5]*(cpro_gradp[c_id][2]));
          vel[c_id][1] =   vel[c_id][1] * rhok1drhok
                         - unsrom*(  dttens[c_id][3]*(cpro_gradp[c_id][0])
                                   + dttens[c_id][1]*(cpro_gradp[c_id][1])
                                   + dttens[c_id][4]*(cpro_gradp[c_id][2]));
          vel[c_id][2] =   vel[c_id][2] * rhok1drhok
                         - unsrom*(  dttens[c_id][5]*(cpro_gradp[c_id][0])
                                   + dttens[c_id][4]*(cpro_gradp[c_id][1])
                                   + dttens[c_id][2]*(cpro_gradp[c_id][2]));
        });
      }

    } /* vp_param->iphydr */

    if (gradp != nullptr)
      CS_FREE_HD(gradp);
  }

  /* RT0 update from the mass fluxes */
  else { /* vp_param->irevmc != 0) */

    cs_arrays_set_value<cs_real_t, 1>(3*n_cells_ext, 0.,(cs_real_t *)vel);

    /* vel = 1 / (rho Vol) SUM mass_flux (X_f - X_i) */
    if (vof_param->vof_model == 0) {

      const cs_real_t *cell_f_vol = mq->cell_f_vol;

      ctx.parallel_for_i_faces(m, [=] CS_F_HOST_DEVICE (cs_lnum_t  face_id) {
        const cs_lnum_t c_id1 = i_face_cells[face_id][0];
        const cs_lnum_t c_id2 = i_face_cells[face_id][1];

        cs_real_t vol_fl_drhovol1 = 0,  vol_fl_drhovol2 = 0;

        /* If it is not a solid cell */
        const int ind1 = has_disable_flag * c_id1;
        const int c_act1 = (1 - (has_disable_flag * c_disable_flag[ind1]));
        if (c_act1 == 1)
          vol_fl_drhovol1 = imasfl[face_id] / (crom[c_id1]*cell_f_vol[c_id1]);

        /* If it is not a solid cell */
        const int ind2 = has_disable_flag * c_id2;
        const int c_act2 = (1 - (has_disable_flag * c_disable_flag[ind2]));
        if (c_act2 == 1)
          vol_fl_drhovol2 = imasfl[face_id] / (crom[c_id2]*cell_f_vol[c_id2]);

        cs_real_t flui[3], fluj[3];
        for (cs_lnum_t k = 0; k < 3; k++) {
          flui[k] =  vol_fl_drhovol1*(i_face_cog[face_id][k]-cell_cen[c_id1][k]);
          fluj[k] = -vol_fl_drhovol2*(i_face_cog[face_id][k]-cell_cen[c_id2][k]);
        }

        if (c_id1 < n_cells)
          cs_dispatch_sum<3>(vel[c_id1], flui, i_sum_type);
        if (c_id2 < n_cells)
          cs_dispatch_sum<3>(vel[c_id2], fluj, i_sum_type);

      });

      ctx.parallel_for_b_faces(m, [=] CS_F_HOST_DEVICE (cs_lnum_t  face_id) {
        const cs_lnum_t c_id1 = b_face_cells[face_id];

        cs_real_t vol_fl_drhovol1 = 0;
        /* If it is not a solid cell */
        const int ind1 = has_disable_flag * c_id1;
        const int c_act1 = (1 - (has_disable_flag * c_disable_flag[ind1]));
        if (c_act1 == 1)
          vol_fl_drhovol1 = bmasfl[face_id]/(crom[c_id1]*cell_f_vol[c_id1]);

        cs_real_t flui[3];
        for (cs_lnum_t k = 0; k < 3; k++)
          flui[k] = vol_fl_drhovol1*(b_face_cog[face_id][k]-cell_cen[c_id1][k]);

        cs_dispatch_sum<3>(vel[c_id1], flui, b_sum_type);

      });
    }

    else { /* if (vof_param->vof_model > 1) */

      const cs_real_t *cell_f_vol = mq->cell_f_vol;

      /* Id of the volume flux */
      const int kimasf = cs_field_key_id("inner_mass_flux_id");
      const int kbmasf = cs_field_key_id("boundary_mass_flux_id");
      const int ivolfl_id
        = cs_field_get_key_int(cs_field_by_name("void_fraction"), kimasf);
      const int bvolfl_id
        = cs_field_get_key_int(cs_field_by_name("void_fraction"), kbmasf);
      const cs_real_t *ivolfl = cs_field_by_id(ivolfl_id)->val;
      const cs_real_t *bvolfl = cs_field_by_id(bvolfl_id)->val;

      ctx.parallel_for_i_faces(m, [=] CS_F_HOST_DEVICE (cs_lnum_t  face_id) {
        const cs_lnum_t c_id1 = i_face_cells[face_id][0];
        const cs_lnum_t c_id2 = i_face_cells[face_id][1];

        cs_real_t vol_fl_drhovol1 = 0, vol_fl_drhovol2 = 0;

        /* If it is not a solid cell */
        const int ind1 = has_disable_flag * c_id1;
        const int c_act1 = (1 - (has_disable_flag * c_disable_flag[ind1]));
        if (c_act1 == 1)
          vol_fl_drhovol1 = ivolfl[face_id] / cell_f_vol[c_id1];

        /* If it is not a solid cell */
        const int ind2 = has_disable_flag * c_id2;
        const int c_act2 = (1 - (has_disable_flag * c_disable_flag[ind2]));
        if (c_act2 == 1)
          vol_fl_drhovol2 = ivolfl[face_id] / cell_f_vol[c_id2];

        cs_real_t flui[3], fluj[3];
        for (cs_lnum_t k = 0; k < 3; k++) {
          flui[k] =  vol_fl_drhovol1*(i_face_cog[face_id][k]-cell_cen[c_id1][k]);
          fluj[k] = -vol_fl_drhovol2*(i_face_cog[face_id][k]-cell_cen[c_id2][k]);
        }

        if (c_id1 < n_cells)
          cs_dispatch_sum<3>(vel[c_id1], flui, i_sum_type);
        if (c_id2 < n_cells)
          cs_dispatch_sum<3>(vel[c_id2], fluj, i_sum_type);

      });

      ctx.parallel_for_b_faces(m, [=] CS_F_HOST_DEVICE (cs_lnum_t  face_id) {
        const cs_lnum_t c_id1 = b_face_cells[face_id];

        cs_real_t vol_fl_drhovol1 = 0;
        /* If it is not a solid cell */
        const int ind1 = has_disable_flag * c_id1;
        const int c_act1 = (1 - (has_disable_flag * c_disable_flag[ind1]));
        if (c_act1 == 1)
          vol_fl_drhovol1 = bvolfl[face_id] / cell_f_vol[c_id1];

        cs_real_t flui[3];
        for (cs_lnum_t k = 0; k < 3; k++)
          flui[k] = vol_fl_drhovol1*(b_face_cog[face_id][k]-cell_cen[c_id1][k]);

        cs_dispatch_sum<3>(vel[c_id1], flui, b_sum_type);
      });

    }
  } /* vp_param->irevmc */

  ctx.wait(); // needed for the following synchronization

  cs_mesh_sync_var_vect((cs_real_t *)vel);

  if (vp_param->iphydr == 1) {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      const int ind = has_disable_flag * c_id;
      const int is_active = (1 - (has_disable_flag * c_disable_flag[ind]));

      for (cs_lnum_t ii = 0; ii < 3; ii++)
        frcxt[c_id][ii] =   frcxt[c_id][ii]*is_active
                          + dfrcxt[c_id][ii];
    });

    ctx.wait(); // needed for the following synchronization

    cs_mesh_sync_var_vect((cs_real_t *)frcxt);
  }

}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Print norms of density, velocity and pressure in listing.
 *
 * \param[in]  m         pointer to associated mesh structure
 * \param[in]  mq        pointer to associated mesh quantities structure
 * \param[in]  iterns    sub-iteration count
 * \param[in]  icvrge    convergence indicator
 * \param[in]  crom      density at cells
 * \param[in]  brom      density at boundary faces
 * \param[in]  imasfl    interior face mass flux
 * \param[in]  bmasfl    boundary face mass flux
 * \param[in]  cvar_pr   pressure
 * \param[in]  cvar_vel  velocity
 */
/*----------------------------------------------------------------------------*/

static void
_log_norm(const cs_mesh_t                *m,
          const cs_mesh_quantities_t     *mq,
          int                             iterns,
          int                             icvrge,
          const cs_real_t                 crom[],
          const cs_real_t                 brom[],
          const cs_real_t                 imasfl[],
          const cs_real_t                 bmasfl[],
          const cs_real_t                 cvar_pr[],
          const cs_real_3_t               cvar_vel[])
{
  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_i_faces = m->n_i_faces;
  const cs_lnum_t n_b_faces = m->n_b_faces;

  const cs_lnum_2_t *restrict i_face_cells
    = (const cs_lnum_2_t *restrict)m->i_face_cells;
  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *restrict)m->b_face_cells;

  const cs_real_3_t *cell_cen = (const cs_real_3_t *)mq->cell_cen;
  const cs_real_t *i_face_surf = mq->i_face_surf;
  const cs_real_t *i_f_face_surf = mq->i_f_face_surf;
  const cs_real_t *b_face_surf = mq->b_face_surf;
  const cs_real_t *b_f_face_surf = mq->b_f_face_surf;

  cs_log_printf(CS_LOG_DEFAULT,
                _(" AFTER CONTINUITY PRESSURE\n"
                  " -------------------------\n"));
  cs_real_t rnorm = -1.0, rnormt = -1.0;

  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++)
    rnorm = fmax(rnorm, fabs(cvar_pr[c_id]));
  cs_parall_max(1, CS_REAL_TYPE, &rnorm);

  bft_printf("Max. pressure, %12.4e, (max. absolute value)\n", rnorm);

  rnorm = -1.0;
  cs_lnum_t imax = 1, imaxt = -1;
  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
    const cs_real_t vitnor = cs_math_3_norm(cvar_vel[c_id]);
   if (vitnor >= rnormt) {
     imaxt  = c_id;
     rnormt = vitnor;
   }
  }
  if (rnormt > rnorm) {
    imax = imaxt;
    rnorm = rnormt;
  }

  cs_real_t xyzmax[3] = {cell_cen[imax][0],
                         cell_cen[imax][1],
                         cell_cen[imax][2]};

  cs_parall_max_loc_vals(3, &rnorm, xyzmax);

  bft_printf("Max. velocity, %12.4e, in, %11.3e, %11.3e, %11.3e\n",
             rnorm, xyzmax[0], xyzmax[1], xyzmax[2]);

  cs_lnum_t imin = 1, imint = 1;
  rnorm = cs_math_3_norm(cvar_vel[0]);
  rnormt = rnorm;
  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
    const cs_real_t vitnor = cs_math_3_norm(cvar_vel[c_id]);
     if (vitnor <= rnormt) {
       imint  = c_id;
       rnormt = vitnor;
     }
  }
  if (rnormt < rnorm) {
    imin = imint;
    rnorm = rnormt;
  }

  cs_real_t xyzmin[3] = {cell_cen[imin][0],
                         cell_cen[imin][1],
                         cell_cen[imin][2]};

  cs_parall_min_loc_vals(3, &rnorm, xyzmin);

  bft_printf("Min. velocity,%12.4e, in, %11.3e, %11.3e, %11.3e\n",
             rnorm, xyzmin[0], xyzmin[1], xyzmin[2]);

  const cs_real_t *ivolfl = nullptr, *bvolfl = nullptr;

  const int iporos = cs_glob_porous_model;
  cs_real_t *porosi = nullptr;

  /* With porosity */
  if (iporos > 0) {
    porosi = CS_F_(poro)->val;
    cs_mesh_sync_var_scal(porosi);
  }

  if (cs_glob_vof_parameters->vof_model > 0) {
    const int kimasf = cs_field_key_id("inner_mass_flux_id");
    const int kbmasf = cs_field_key_id("boundary_mass_flux_id");
    const int ivolfl_id
      = cs_field_get_key_int(cs_field_by_name("void_fraction"), kimasf);
    const int bvolfl_id
      = cs_field_get_key_int(cs_field_by_name("void_fraction"), kbmasf);

    ivolfl = cs_field_by_id(ivolfl_id)->val;
    bvolfl = cs_field_by_id(bvolfl_id)->val;
  }

  cs_real_t rnormi = cs_math_big_r;
  cs_real_t rnorma = -cs_math_big_r;

  for (cs_lnum_t face_id = 0; face_id < n_i_faces; face_id++) {
    const cs_lnum_t c_id1 = i_face_cells[face_id][0];
    const cs_lnum_t c_id2 = i_face_cells[face_id][1];
    cs_real_t rhom;
    if (iporos == 1 || iporos == 2)
      rhom = (porosi[c_id1]*crom[c_id1] + porosi[c_id2]*crom[c_id2])*0.5;
    else
      rhom = (crom[c_id1] + crom[c_id2])*0.5;
    /* Deal with null fluid section */
    rnorm = 0.;
    if (i_f_face_surf[face_id] / i_face_surf[face_id] > cs_math_epzero) {
      rnorm = fabs(imasfl[face_id]) / (i_f_face_surf[face_id]*rhom);
      if (cs_glob_vof_parameters->vof_model > 0)
        rnorm = fabs(ivolfl[face_id]) / i_f_face_surf[face_id];
    }
    rnorma = cs_math_fmax(rnorma, rnorm);
    rnormi = cs_math_fmin(rnormi, rnorm);
  }
  cs_parall_min(1, CS_REAL_TYPE, &rnormi);
  cs_parall_max(1, CS_REAL_TYPE, &rnorma);

  bft_printf(" Max. velocity at interior faces %12.4e; min. %12.4e\n",
             rnorma, rnormi);

  rnormi = cs_math_big_r;
  rnorma = -cs_math_big_r;

  for (cs_lnum_t face_id = 0; face_id < n_b_faces; face_id++) {
    if (bvolfl != nullptr) {
      /*  Deal with null fluid section */
      rnorm = 0;
      if (b_f_face_surf[face_id] / b_face_surf[face_id] > cs_math_epzero)
        rnorm = bvolfl[face_id] / (b_f_face_surf[face_id]);
    }
    else {
      const cs_lnum_t c_id = b_face_cells[face_id];
      if ((iporos == 1) || (iporos == 2))
        rnorm = bmasfl[face_id]
               / (b_face_surf[face_id]*brom[face_id]*porosi[c_id]);
      else {
      /* Deal with null fluid section */
        rnorm = 0;
        if (mq->b_f_face_surf[face_id]/mq->b_face_surf[face_id] > cs_math_epzero)
          rnorm = bmasfl[face_id]/(mq->b_f_face_surf[face_id]*brom[face_id]);
      }
    }
    rnorma = fmax(rnorma, rnorm);
    rnormi = fmin(rnormi, rnorm);
  }
  cs_parall_min(1, CS_REAL_TYPE, &rnormi);
  cs_parall_max(1, CS_REAL_TYPE, &rnorma);

  bft_printf(" Max. velocity at boundary faces %12.4e; min. %12.4e\n",
             rnorma, rnormi);

  rnorm = cs_sum(n_b_faces, bmasfl);
  cs_parall_sum(1, CS_REAL_TYPE, &rnorm);

  bft_printf(" Mass balance  at boundary: %14.6e\n", rnorm);
  bft_printf(" ----------------------------------------\n");

  const cs_velocity_pressure_param_t *vp_param = cs_glob_velocity_pressure_param;

  if (vp_param->nterup > 1) {
    if (icvrge == 0) {
      bft_printf(" Fixed point for velocity-pressure coupling at iteration: "
                 "%d\n", iterns);
      bft_printf("   norm = %12.4e, norm 0 = %12.4e, toler = %12.4e\n",
                 vp_param->xnrmu, vp_param->xnrmu0, vp_param->epsup);
      bft_printf(" ------------------------------------------------------\n");
      if (iterns == vp_param->nterup) {
        bft_printf(" Non convergence of fixed point for velocity-pressure "
                   "coupling"
                   " ------------------------------------------------------\n");
      }
    }
    else {
      bft_printf(" Fixed point convergence at iteration %d", iterns);
      bft_printf("   norm = %12.4e, norm 0 = %12.4e, toler = %12.4e\n",
                 vp_param->xnrmu, vp_param->xnrmu0, vp_param->epsup);
      bft_printf(" ------------------------------------------------------\n");
    }
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Print norms of density, velocity and pressure in listing.
 *
 * \param[in]  m         pointer to associated mesh structure
 * \param[in]  mq        pointer to associated mesh quantities structure
 * \param[in]  iterns    sub-iteration count
 * \param[in]  icvrge    convergence indicator
 * \param[in]  crom      density at cells
 * \param[in]  brom      density at boundary faces
 * \param[in]  imasfl    interior face mass flux
 * \param[in]  bmasfl    boundary face mass flux
 * \param[in]  cvar_pr   pressure
 * \param[in]  cvar_vel  velocity
 */
/*----------------------------------------------------------------------------*/

static void
_resize_non_interleaved_cell_arrays(const cs_mesh_t    *m,
                                    cs_lnum_t           n_sub,
                                    cs_real_t         **array)
{
  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;

  cs_dispatch_context ctx;

  cs_real_t *buffer = nullptr;

  CS_MALLOC_HD(buffer, n_sub*n_cells, cs_real_t, cs_alloc_mode);
  for (cs_lnum_t i = 0; i < n_sub; i++) {
    cs_array_copy<cs_real_t>(n_cells, *array + i*n_cells_ext, buffer + i*n_cells);
  }

  ctx.wait();

  CS_REALLOC_HD(*array, n_sub*n_cells_ext, cs_real_t, cs_alloc_mode);

  for (cs_lnum_t i = 0; i < n_sub; i++) {
    cs_real_t *src = buffer + i*n_cells;
    cs_real_t *dst = *array + i*n_cells_ext;
    cs_array_copy<cs_real_t>(n_cells, src, dst);
    ctx.wait();
    cs_mesh_sync_var_scal(dst);
  }

  CS_FREE_HD(buffer);
}

/*----------------------------------------------------------------------------*/
/*!
  * \brief Velocity prediction step of the Navier-Stokes equations for
  *        incompressible or slightly compressible flows.
  *
  * - At the first call, the predicted velocities are computed as well
  *   as an estimator on the predicted velocity.
  *
  * - At the second call, a global estimator on Navier Stokes is computed.
  *   This second call is done after the correction step
  *   (\ref cs_pressure_correction).
  *
  * Please refer to the
  * <a href="../../theory.pdf#cs_velocity_prediction"><b>cs_velocity_prediction</b></b></a>
  * section of the theory guide for more informations.
  *
  * \param[in]       iappel        call number (1 or 2)
  * \param[in]       iterns        index of the iteration on Navier-Stokes
  * \param[in]       dt            time step (per cell)
  * \param[in]       vel           velocity
  * \param[in]       vela          velocity at the previous time step
  * \param[in]       velk          velocity at the previous sub iteration (or vela)
  * \param[in,out]   da_uu         velocity matrix
  * \param[in]       bc_coeffs_v   boundary condition structure for the variable
  * \param[in]       ckupdc        head loss coefficients, if present
  * \param[in]       frcxt         external forces making hydrostatic pressure
  * \param[in]       trava         working array for the velocity-pressure coupling
  * \param[out]      dfrcxt        variation of the external forces
  *                                making the hydrostatic pressure
  * \param[in]       grdphd        hydrostatic pressure gradient to handle the
  *                                imbalance between the pressure gradient and
  *                                gravity source term
  * \param[in]       gxyz          gravity
  * \param[in, out]  dttens        non scalar time step in case of
  *                                velocity pressure coupling
  * \param[in, out]  trav          right hand side for the normalizing
  *                                the residual
  * \param[in, out]  viscf         visc*surface/dist aux faces internes
  * \param[in, out]  viscb         visc*surface/dist aux faces de bord
  * \param[in, out]  viscfi        same as viscf for increments
  * \param[in, out]  viscbi        same as viscb for increments
  * \param[in, out]  secvif        secondary viscosity at interior faces
  * \param[in, out]  secvib        secondary viscosity at boundary faces
  */
/*----------------------------------------------------------------------------*/

static void
_velocity_prediction(const cs_mesh_t             *m,
                     const cs_mesh_quantities_t  *mq,
                     int                          iappel,
                     int                          iterns,
                     const cs_real_t              dt[],
                     cs_real_t                    vel[][3],
                     cs_real_t                    vela[][3],
                     cs_real_t                    velk[][3],
                     cs_real_t                    da_uu[][6],
                     cs_field_bc_coeffs_t        *bc_coeffs_v,
                     const cs_real_t              ckupdc[][6],
                     cs_real_t                    frcxt[][3],
                     cs_real_t                    grdphd[][3],
                     const cs_real_t              gxyz[3],
                     const cs_real_t              xyzp0[3],
                     cs_real_t                    trava[][3],
                     cs_real_t                    dfrcxt[][3],
                     cs_real_t                    dttens[][6],
                     cs_real_t                    trav[][3],
                     cs_real_t                    viscf[],
                     cs_real_t                    viscb[],
                     cs_real_t                    viscfi[],
                     cs_real_t                    viscbi[],
                     cs_real_t                    secvif[],
                     cs_real_t                    secvib[])
{
  cs_real_33_t *cofbfv = (cs_real_33_t *)bc_coeffs_v->bf;
  cs_lnum_t n_cells = m->n_cells;
  cs_lnum_t n_i_faces = m->n_i_faces;
  cs_lnum_t n_b_faces = m->n_b_faces;
  cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;

  const cs_lnum_t *b_face_cells = m->b_face_cells;

  const cs_real_t *cell_f_vol = mq->cell_f_vol;
  const cs_real_3_t *restrict diipb = (const cs_real_3_t *restrict)mq->diipb;

  const cs_real_3_t  *restrict b_face_normal
    = (const cs_real_3_t  *restrict) mq->b_face_normal;
  int has_disable_flag = mq->has_disable_flag;
  int *c_disable_flag = mq->c_disable_flag;

  const cs_time_step_t *ts = cs_glob_time_step;
  const cs_time_step_options_t  *tso = cs_glob_time_step_options;
  const cs_fluid_properties_t *fp = cs_glob_fluid_properties;
  const cs_vof_parameters_t *vof_param = cs_glob_vof_parameters;
  const cs_velocity_pressure_model_t
    *vp_model = cs_glob_velocity_pressure_model;
  const cs_velocity_pressure_param_t *vp_param = cs_glob_velocity_pressure_param;

  cs_equation_param_t *eqp_u
    = cs_field_get_equation_param(CS_F_(vel));

  const cs_equation_param_t *eqp_p
    = cs_field_get_equation_param_const(CS_F_(p));

  const int kimasf = cs_field_key_id("inner_mass_flux_id");
  const int kbmasf = cs_field_key_id("boundary_mass_flux_id");
  const int iflmas = cs_field_get_key_int(CS_F_(vel), kimasf);
  const int iflmab = cs_field_get_key_int(CS_F_(vel), kbmasf);

  cs_real_t *imasfl = cs_field_by_id(iflmas)->val;
  cs_real_t *bmasfl = cs_field_by_id(iflmab)->val;

  const cs_real_t ro0 = fp->ro0;
  const cs_real_t pred0 = fp->pred0;

  /* Pointers to properties
   * Density at time n+1,iteration iterns+1 */
  cs_real_t *crom_eos = CS_F_(rho)->val;
  cs_real_t *brom_eos = CS_F_(rho_b)->val;

  /* Density at time (n) */
  cs_real_t *croma = crom_eos;
  cs_real_t *broma = brom_eos;
  if (fp->irovar == 1) {
    croma = CS_F_(rho)->val_pre;
    broma = CS_F_(rho_b)->val_pre;
  }

  /* Density at time (n-1) if needed */
  cs_real_t *cromaa = nullptr;
  if (   vp_model->idilat > 1
      || vof_param->vof_model > 0
      || cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 3
      || fp->irovar == 1)
    cromaa = CS_F_(rho)->vals[2];

  /* Add Rusanov */
  cs_real_t *ipro_rusanov = nullptr;
  if (cs_glob_turb_rans_model->irijnu == 2)
    ipro_rusanov = cs_field_by_name("i_rusanov_diff")->val;

  /* Density for the unsteady term (at time n);
     by default (constant or weakly variable density), set to
     density as defined by equations of state. */
  cs_real_t *pcrom = crom_eos;

  if (fp->irovar == 1) {
    /* Compressible algorithm (mass equation is already solved)
     * or Low Mach compressible algos with mass flux prediction */
    if (   (   cs_glob_physical_model_flag[CS_COMPRESSIBLE] >=0
            && cs_glob_physical_model_flag[CS_COMPRESSIBLE] != 3)
        || (vp_model->idilat > 1 && vp_param->ipredfl == 1))
      pcrom = croma;

    /* VOF algorithm and Low Mach compressible algos: density at time n-1 */
    else if (   vp_model->idilat > 1
             || vof_param->vof_model > 0
             || cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 3) {
      if (vp_param->itpcol == 0 && iterns == 1)
        pcrom = cromaa;
      else
        pcrom = croma;
    }
  }

  /* Density for other terms such as buoyancy term
     (default for 1st order in time) */
  cs_real_t *crom = crom_eos, *brom = brom_eos;

  if (eqp_u->theta < 1.0) {   /* 2nd order in time */
   /* map the density pointer:
    * 1/4(n-1) + 1/2(n) + 1/4(n+1)
    * here replaced by (n) */
    crom = croma;
    brom = broma;
  }

  cs_dispatch_context ctx;

  /* Interpolation of rho^n-1/2 (stored in pcrom)
   * Interpolation of the mass flux at (n+1/2)
   * NB: the mass flux (n+1) is overwritten because not used after.
   * The mass flux for (n->n+1) will be recomputed in cs_pressure_correction
   * FIXME irovar=1 and if dt varies, use theta(rho) = theta(u)*... */

  cs_real_t *cproa_rho_tc = nullptr;
  if (   (eqp_u->theta < 1.0) && (iappel == 1)
      && (iterns > 1) && (vp_param->itpcol == 0)) {
    CS_MALLOC_HD(cproa_rho_tc, n_cells_ext, cs_real_t, cs_alloc_mode);

    /* Pointer to the previous mass fluxes */
    cs_real_t *imasfl_prev = cs_field_by_id(iflmas)->val_pre;
    cs_real_t *bmasfl_prev = cs_field_by_id(iflmab)->val_pre;

    const cs_real_t theta = eqp_u->theta;

    if (fp->irovar == 1) {
      /* remap the density pointer: n-1/2 */
      ctx.parallel_for(n_cells_ext, [=] CS_F_HOST_DEVICE (cs_lnum_t  c_id) {
        cproa_rho_tc[c_id] =          theta  * croma[c_id]
                             + (1.0 - theta) * cromaa[c_id];
      });

      pcrom = cproa_rho_tc;
    }

    /* Inner mass flux interpolation: n-1/2->n+1/2 */
    ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t  f_id) {
      imasfl[f_id] =          theta  * imasfl[f_id]
                     + (1.0 - theta) * imasfl_prev[f_id];
    });

    /* Boundary mass flux interpolation: n-1/2->n+1/2 */
    ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t  f_id) {
      bmasfl[f_id] =          theta  * bmasfl[f_id]
                     + (1.0 - theta) * bmasfl_prev[f_id];
    });
  }

  cs_real_6_t *viscce = nullptr;
  if (eqp_u->idften & CS_ANISOTROPIC_LEFT_DIFFUSION)
    CS_MALLOC_HD(viscce, n_cells_ext, cs_real_6_t, cs_alloc_mode);

  cs_field_t *iespre = cs_field_by_name_try("est_error_pre_2");

  cs_real_t *cvar_pr = nullptr;
  cs_real_t *cvara_k = nullptr;

  cs_field_t *iforbr = cs_field_by_name_try("boundary_forces");

  if ((iforbr != nullptr && iterns == 1) || (vof_param->vof_model > 0))
    cvar_pr = CS_F_(p)->val;

  if (   iterns == 1
      && iforbr != nullptr
      && cs_glob_turb_rans_model->igrhok == 1
      && (   cs_glob_turb_model->itytur == 2
          || cs_glob_turb_model->itytur == 5
          || cs_glob_turb_model->iturb == CS_TURB_K_OMEGA)) {
    if (iappel == 2)
      cvara_k = CS_F_(k)->val;
    else
      cvara_k = CS_F_(k)->val_pre;
  }

  cs_real_3_t *forbr = nullptr;
  if (iforbr != nullptr && iterns == 1)
    forbr = (cs_real_3_t *)iforbr->val;

  cs_real_3_t *c_st_vel = nullptr;
  const cs_real_t thets = cs_glob_time_scheme->thetsn;

  if (cs_glob_time_scheme->isno2t > 0) {
    int kstprv = cs_field_key_id("source_term_prev_id");
    int istprv = cs_field_get_key_int(CS_F_(vel), kstprv);
    if (istprv > -1)
      c_st_vel = (cs_real_3_t *)cs_field_by_id(istprv)->val;
  }

  /* Get user source terms */
  cs_field_t *f = cs_field_by_name_try("velocity_source_term_exp");
  cs_real_3_t *loctsexp = nullptr, *tsexp = nullptr;
  if (f != nullptr)
    tsexp = (cs_real_3_t *)f->val;
  else {
    CS_MALLOC_HD(loctsexp, n_cells_ext, cs_real_3_t, cs_alloc_mode);
    tsexp = loctsexp;
  }

  f = cs_field_by_name_try("velocity_source_term_imp");
  cs_real_33_t *loctsimp = nullptr, *tsimp = nullptr;
  if (f != nullptr)
    tsimp = (cs_real_33_t *)f->val;
  else {
    CS_MALLOC_HD(loctsimp, n_cells_ext, cs_real_33_t, cs_alloc_mode);
    tsimp = loctsimp;
  }

  ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
    for (cs_lnum_t i = 0; i < 3; i++) {
      tsexp[c_id][i] = 0.;
      for (cs_lnum_t j = 0; j < 3; j++)
        tsimp[c_id][i][j] = 0.;
    }
  });

  /* The computation of explicit and implicit source terms is performed
   * at the first iteration only.
   * If iphydr=1 or if we have buoyant scalars
   * then we need to update source terms */

  cs_gui_momentum_source_terms(vel, tsexp, tsimp);

  cs_user_source_terms(cs_glob_domain,
                       CS_F_(vel)->id,
                       (cs_real_t *)tsexp,
                       (cs_real_t *)tsimp);

  if (cs_glob_porous_model == 3)
    cs_immersed_boundary_wall_functions(CS_F_(vel)->id,
                                        (cs_real_t *)tsexp,
                                        (cs_real_t *)tsimp);

  if (cs_fan_n_fans() > 0) {
    if (ts->nt_cur == ts->nt_prev+1)
      cs_fan_compute_flows(cs_glob_mesh,
                           cs_glob_mesh_quantities,
                           imasfl,
                           bmasfl,
                           crom,
                           brom);
    cs_fan_compute_force(mq, tsexp);
  }

  if (cs_glob_physical_model_flag[CS_PHYSICAL_MODEL_FLAG] > 0) {
    if (cs_glob_physical_model_flag[CS_COOLING_TOWERS] > 0){
      cs_ctwr_source_term(CS_F_(vel)->id,
                          (cs_real_t *)tsexp,
                          (cs_real_t *)tsimp);
    }
  }

  /* Skip first time step after restart if previous values have not been read. */
  if (eqp_u->ibdtso < 0)
    eqp_u->ibdtso = -eqp_u->ibdtso;

  /* Nudging towards optimal interpolation for velocity */
  if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] > CS_ATMO_OFF) {
    const int kopint = cs_field_key_id_try("opt_interp_id");
    const int f_oi_id = cs_field_get_key_int(CS_F_(vel), kopint);
    if (f_oi_id > -1)
      cs_at_data_assim_source_term(CS_F_(vel)->id,
                                   (cs_real_t *)tsexp,
                                   (cs_real_t *)tsimp);

    if (cs_glob_atmo_option->open_bcs_treatment > 0)
      cs_at_source_term_for_inlet(tsexp);
  }

  /* Coupling between two code_saturne instances */
  if (cs_sat_coupling_n_couplings() > 0)
    cs_sat_coupling_exchange_at_cells(CS_F_(vel),
                                      (cs_real_t*)tsexp,
                                      (cs_real_t*)tsimp);

  if (eqp_u->ibdtso > 1 && ts->nt_cur > ts->nt_ini
      && (   tso->idtvar == CS_TIME_STEP_CONSTANT
          || tso->idtvar == CS_TIME_STEP_ADAPTIVE))
    /* TODO: remove test on ntcabs and implemente a "proper" condition for
     * initialization. */
    cs_backward_differentiation_in_time(CS_F_(vel),
                                        (cs_real_t *)tsexp,
                                        (cs_real_t *)tsimp);

  /* Potential forces (pressure gradient and gravity)
     ================================================ */

  /* Pressure gradient */
  cs_real_3_t *grad = nullptr, *cpro_gradp = nullptr;
  f = cs_field_by_name_try("algo:gradient_pressure");
  if (f != nullptr)
    cpro_gradp = (cs_real_3_t *)f->val;
  else {
    CS_MALLOC_HD(grad, n_cells_ext, cs_real_3_t, cs_alloc_mode);
    cpro_gradp = grad;
  }

  cs_real_t *cpro_rho_mass = nullptr;
  cs_real_t *wgrec_crom = nullptr, *cpro_rho_tc = nullptr;

  /* Namely for the VOF algorithm: consistency of the gradient
   * with the diffusive flux scheme of the correction step */
  if (eqp_p->iwgrec == 1) {

    /* retrieve density used in diffusive flux scheme (correction step) */
    if (   fp->irovar == 1
        && (   vp_model->idilat > 1
            || vof_param->vof_model > 0
            || cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 3)) {

      cpro_rho_mass = cs_field_by_name("density_mass")->val;

      /* Time interpolated density */
      if (eqp_u->theta < 1.0 && iterns > 1) {
        cs_real_t theta = eqp_u->theta;
        CS_MALLOC_HD(cpro_rho_tc, n_cells_ext, cs_real_t, cs_alloc_mode);
        ctx.parallel_for(n_cells_ext, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          cpro_rho_tc[c_id] =          theta  * cpro_rho_mass[c_id]
                              + (1.0 - theta) * croma[c_id];
        });
        wgrec_crom = cpro_rho_tc;
      }
      else
        wgrec_crom = cpro_rho_mass;
    }

    /* Weakly variable density algo. (idilat <=1) or constant density */
    else
      wgrec_crom = crom_eos;

    /* Id weighting field for gradient */
    const int kwgrec = cs_field_key_id_try("gradient_weighting_id");
    const int iflwgr = cs_field_get_key_int(CS_F_(p), kwgrec);
    cs_field_t *f_g = cs_field_by_id(iflwgr);
    cs_real_6_t *cpro_wgrec_v = nullptr;
    cs_real_t *cpro_wgrec_s = nullptr;
    if (f_g->dim > 1) {
      cpro_wgrec_v = (cs_real_6_t *)f_g->val;
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          cpro_wgrec_v[c_id][ii] = dt[c_id] / wgrec_crom[c_id];

        for (cs_lnum_t ii = 3; ii < 6; ii++)
          cpro_wgrec_v[c_id][ii] = 0;
      });
      ctx.wait();
      cs_mesh_sync_var_sym_tens(cpro_wgrec_v);
    }
    else {
      cpro_wgrec_s = f_g->val;
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        cpro_wgrec_s[c_id] = dt[c_id] / wgrec_crom[c_id];
      });
      ctx.wait();
      cs_mesh_sync_var_scal(cpro_wgrec_s);
    }
    CS_FREE_HD(cpro_rho_tc);
  }

  cs_gradient_porosity_balance(1);

  /* Pressure gradient */
  if (cs_glob_velocity_pressure_model->iprcdo == 0)
    cs_field_gradient_potential(CS_F_(p),
                                0, /* iprev */
                                1, /* inc */
                                vp_param->iphydr,
                                frcxt,
                                cpro_gradp);

  const cs_real_3_t *restrict cdgfbo
      = (const cs_real_3_t *restrict)mq->b_face_cog;

  /* Compute stress at walls (part 2/5), if required.
   * Face pressure is computed at face and computed as in gradient
   * reconstruction, the transformed into total pressure.
   * We restrict this to the first iteration (for simplicity relatively
   * to the part in cs_boundary_condition_set_coeffs, outside the loop) */

  if (forbr != nullptr && iterns == 1) {
    const cs_real_t *coefa_p = CS_F_(p)->bc_coeffs->a;
    const cs_real_t *coefb_p = CS_F_(p)->bc_coeffs->b;

    ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      const cs_lnum_t c_id = b_face_cells[f_id];
      const cs_real_t pip =   cvar_pr[c_id]
                            + cs_math_3_dot_product(diipb[f_id],
                                                    cpro_gradp[c_id]);

      cs_real_t pfac = coefa_p[f_id] + coefb_p[f_id]*pip;
      pfac +=   ro0 * cs_math_3_distance_dot_product(xyzp0,
                                                     cdgfbo[f_id],
                                                     gxyz)
              - pred0;

      for (cs_lnum_t isou = 0; isou < 3; isou++)
        forbr[f_id][isou] += pfac*b_face_normal[f_id][isou];
    });
  }

  if (iappel == 1)
    /* Initialization
     * NB: at the second call, trav contains the temporal increment */
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      for (cs_lnum_t i = 0; i < 3; i++)
        trav[c_id][i] = 0.;
    });

  /* FIXME : "rho g" will be second order only if extrapolated */

  if (vp_param->iphydr == 1) {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        trav[c_id][ii] +=   (frcxt[c_id][ii] - cpro_gradp[c_id][ii])
                          * cell_f_vol[c_id];
    });
  }
  else if (vp_param->iphydr == 2) {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      const cs_real_t rom = crom[c_id];
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        trav[c_id][ii] +=   (  rom*gxyz[ii] - grdphd[c_id][ii]
                             - cpro_gradp[c_id][ii])
                          * cell_f_vol[c_id];
    });
  }
  else if (cs_glob_physical_model_flag[CS_COMPRESSIBLE] >= 0) {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      const cs_real_t rom = crom[c_id];
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        trav[c_id][ii] += (rom*gxyz[ii] - cpro_gradp[c_id][ii])*cell_f_vol[c_id];
    });
  }
  /* Boussinesq approximation */
  else if (vp_model->idilat == 0) {

    /* FIXME make it dependant on the scalar and use coupled_with_vel_p field */
    const cs_real_t *cvar_t = cs_thermal_model_field()->val;
    const cs_real_t *cpro_beta = cs_field_by_name("thermal_expansion")->val;

    /* Delta rho = - rho_0 beta (T-T0) */
    cs_real_t tref = fp->t0;
    /* for atmospheric flows, variable is potential temperature */
    if (cs_glob_physical_model_flag[CS_ATMOSPHERIC] > CS_ATMO_CONSTANT_DENSITY) {
      const cs_real_t rscp = fp->r_pg_cnst/fp->cp0;
      tref = fp->t0*pow(cs_glob_atmo_constants->ps/fp->p0, rscp);
    }

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      const cs_real_t drom = -crom[c_id]*cpro_beta[c_id]*(cvar_t[c_id] - tref);
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        trav[c_id][ii] +=   (drom*gxyz[ii] - cpro_gradp[c_id][ii])
                          * cell_f_vol[c_id];
    });

  }
  else {
    /* 2nd order */
    if (cs_glob_time_scheme->time_order == 2 && vp_param->itpcol == 1) {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        cs_real_t drom = (1.5*croma[c_id] - 0.5*cromaa[c_id] - ro0);
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          trav[c_id][ii] +=   (drom*gxyz[ii] - cpro_gradp[c_id][ii] )
                            * cell_f_vol[c_id];
      });
    }

    /* 1st order */
    else {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        cs_real_t drom = (crom[c_id] - ro0);
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          trav[c_id][ii] +=   (drom*gxyz[ii] - cpro_gradp[c_id][ii] )
                            * cell_f_vol[c_id];
      });
    }
  }

  CS_FREE_HD(grad);

  /* For iappel = 1 (ie standard call without estimators)
   * trav gathers the source terms which will be recalculated
   * to all iterations on navsto
   * If we don't iterate on navsto and we don't extrapolate the
   * source terms, trav contains all source terms
   * until failover in smbr
   * At this level, trav contains -grad P and rho g
   * P is assumed to be taken at n+1/2
   * rho is possibly interpolated at n+1/2 */

  /* Initialize trava array and source terms at the first call (iterns=1)

   *  trava contains all source terms needed from the first sub iteration
   *   (iterns=1) for the other iterations.
   *  When there is only one iteration, we build source terms directly in
   *    the trav array.
   *  Explicit source terms will be used at the next time step in case of
   *    extrapolation (if there is only one or multiple iterations on navtsv) */

  /* At the first iteration on cs_solve_navier_stokes */
  if (iterns == 1) {

    /* If we extrapolate the S.T.: -theta*previous value */
    if (cs_glob_time_scheme->isno2t > 0) {
      if (vp_param->nterup == 1) {
        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
          for (cs_lnum_t ii = 0; ii < 3; ii++) {
            trav[c_id][ii] -= thets*c_st_vel[c_id][ii];
            /* And we initialize the source term to fill it then */
            c_st_vel[c_id][ii] = 0.;
          }
        });
      }
      else {
       ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
         for (cs_lnum_t ii = 0; ii < 3; ii++) {
            trava[c_id][ii] = - thets*c_st_vel[c_id][ii];
            /* And we initialize the source term to fill it then */
            c_st_vel[c_id][ii] = 0.;
         }
        });
      }
      ctx.wait();
    }
    /* If we not extrapolate the ST. */
    else {

      /* If we have many iterationss: trava initialize */
      /* otherwise trava should not exist */
      if (vp_param->nterup > 1)
        ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
         for (cs_lnum_t ii = 0; ii < 3; ii++)
            trava[c_id][ii] = 0.;
        });

    }
  }

  /* Initialization of the implicit terms */

  cs_real_33_t *fimp = nullptr;
  CS_MALLOC_HD(fimp, n_cells_ext, cs_real_33_t, cs_alloc_mode);

  if (iappel == 1 && eqp_u->istat == 1) {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      const cs_real_t fimp_c = pcrom[c_id] / dt[c_id] * cell_f_vol[c_id];
      for (cs_lnum_t ii = 0; ii < 3; ii++) {
        for (cs_lnum_t jj = 0; jj < 3; jj++) {
          if (jj == ii)
            fimp[c_id][ii][ii] = fimp_c;
          else
            fimp[c_id][ii][jj] = 0;
        }
      }
    });
  }
  else {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        for (cs_lnum_t jj = 0; jj < 3; jj++)
          fimp[c_id][ii][jj] = 0.;
    });
  }

  ctx.wait();
  CS_FREE_HD(cproa_rho_tc);

  /* 2/3 rho * grad(k) for k-epsilon ou k-omega
   * Note: we do not take the gradient of (rho k), as this would make
   *       the handling of BC's more complex...
   *
   * It is not clear whether the extrapolation in time is useful.
   *
   * This explicit term is computed once, at the first iteration on
   * cs_solve_navier_stokes: it is saved in a field if it must be extrapolated
   * in time ; it goes into trava if we do not extrapolate or iterate on
   * cs_solve_navier_stokes. */

  if (  (   cs_glob_turb_model->itytur == 2
         || cs_glob_turb_model->itytur == 5
         || cs_glob_turb_model->iturb == CS_TURB_K_OMEGA)
      && cs_glob_turb_rans_model->igrhok == 1 && iterns == 1) {
    cs_real_3_t *grad_k = nullptr;
    BFT_MALLOC(grad_k, n_cells_ext, cs_real_3_t);

    cs_field_gradient_scalar(CS_F_(k), true, 1, grad_k);

    const cs_real_t d2s3 = 2.0/3;
    cs_real_3_t *st_ctrb = nullptr;

    /* If we extrapolate the source terms in time */
    if (cs_glob_time_scheme->isno2t > 0) {
      /* Compute  rho^n grad k^n if rho not extrapolated
       *          rho^n grad k^n if rho     extrapolated */
      st_ctrb = c_st_vel;
    }
    /* If the source terms are not extrapolated in time: trav or trava */
    else {
      if (vp_param->nterup == 1)
        st_ctrb = trav;
      else
        st_ctrb = trava;
    }

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      const cs_real_t romvom = -crom[c_id]*cell_f_vol[c_id]*d2s3;
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        st_ctrb[c_id][ii] += grad_k[c_id][ii] * romvom;
    });

    /* Calculation of wall stresses (part 3/5), if requested */
    if (iforbr != nullptr) {
      const cs_real_t *coefa_k = CS_F_(k)->bc_coeffs->a;
      const cs_real_t *coefb_k = CS_F_(k)->bc_coeffs->b;

      ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
        const cs_lnum_t c_id = b_face_cells[f_id];
        cs_real_t xkb =   cvara_k[c_id]
                        + cs_math_3_dot_product(diipb[f_id],
                                                grad_k[c_id]);

        xkb = coefa_k[f_id] + coefb_k[f_id]*xkb;
        xkb = d2s3 * crom[c_id] * xkb;
        for (cs_lnum_t isou = 0; isou < 3; isou++)
          forbr[f_id][isou] += xkb*b_face_normal[f_id][isou];
      });
    }

    BFT_FREE(grad_k);
  }

  /* Transpose of velocity gradient in the diffusion term
   * These terms are taken into account in cs_balance_vector.
   * We only compute here the secondary viscosity. */

  if (vp_model->ivisse == 1)
    cs_face_viscosity_secondary(secvif, secvib);

  /* Head losses
   * -----------
   * (if iphydr=1 this term has already been taken into account)
   *
   * Remark: icepdc is rebuilt locally, but can be avoided
   * in the future by simply looping over the required zones.
   * This also requires that the "iflow" Lagrangian rentrainment
   * model simply force the base "all cells" zone to head loss
   * type so that it fits in the regular framework.
   */

  cs_lnum_t *icepdc = nullptr;
  cs_lnum_t ncepdc = cs_volume_zone_n_type_cells(CS_VOLUME_ZONE_HEAD_LOSS);

  CS_MALLOC_HD(icepdc, ncepdc, cs_lnum_t, cs_alloc_mode);

  cs_volume_zone_select_type_cells(CS_VOLUME_ZONE_HEAD_LOSS, icepdc);
  if (cs_glob_lagr_reentrained_model->iflow == 1) {
    ctx.parallel_for(ncepdc, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      icepdc[c_id] = c_id;
    });
  }
  ctx.wait();

  /* Explicit part;

   * The diagonal terms are placed in trav or trava,
   * The consideration of velk from the second iteration
   * is done directly in cs_equation_iterative_solve_vector. */

  if (ncepdc > 0 && vp_param->iphydr != 1 && iterns == 1) {

    /* If we have inner iterations, we use trava, otherwise trav */
    if (vp_param->nterup > 1)
      _st_exp_head_loss(ncepdc, icepdc, vela, ckupdc, trava);
    else
      _st_exp_head_loss(ncepdc, icepdc, vela, ckupdc, trav);

  }

  /* Implicit part ;

     At the second call, fimp is not needed anymore */
  if (iappel == 1 && ncepdc > 0) {
    /* The theta-scheme for head loss is the same as the other terms */
    const cs_real_t thetap = eqp_u->theta;

    ctx.parallel_for(ncepdc, [=] CS_F_HOST_DEVICE (cs_lnum_t hl_id) {
      const cs_lnum_t c_id = icepdc[hl_id];
      const cs_real_t romvom = crom[c_id]*cell_f_vol[c_id]*thetap;

      /* Diagonal part */
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        fimp[c_id][ii][ii] += romvom*ckupdc[hl_id][ii];
      /* Extra-diagonal part */
      const cs_real_t cpdc12 = ckupdc[hl_id][3];
      const cs_real_t cpdc23 = ckupdc[hl_id][4];
      const cs_real_t cpdc13 = ckupdc[hl_id][5];

      fimp[c_id][1][0] += romvom*cpdc12;
      fimp[c_id][0][1] += romvom*cpdc12;
      fimp[c_id][2][0] += romvom*cpdc13;
      fimp[c_id][0][2] += romvom*cpdc13;
      fimp[c_id][2][1] += romvom*cpdc23;
      fimp[c_id][1][2] += romvom*cpdc23;
    });
  }

  /* Surface tension force for VoF
     ----------------------------- */

  cs_real_3_t *stf = nullptr;
  if (   cs_glob_vof_parameters->vof_model > 0
      && cs_glob_vof_parameters->sigma_s > 0) {
    BFT_MALLOC(stf, n_cells, cs_real_3_t);
    cs_vof_surface_tension(m, mq, stf);
  }

  /* Coriolis force
     --------------
     (if iphydr=1 then this term is already taken into account) */

  /* Explicit part */
  const int *irotce = cs_turbomachinery_get_cell_rotor_num();
  cs_turbomachinery_model_t iturbo = cs_turbomachinery_get_model();
  if (   (   cs_glob_physical_constants->icorio == 1
          || iturbo == CS_TURBOMACHINERY_FROZEN)
      && (vp_param->iphydr != 1)) {

    /* At first iteration on cs_solve_navier_stokes,
       add the part based on explicit terms */
    if (iterns == 1) {
      cs_real_3_t *trav_p = (vp_param->nterup == 1) ? (cs_real_3_t *)trav
                                                    : (cs_real_3_t *)trava;

      /* Reference frame + turbomachinery frozen rotors rotation */
      if (iturbo == CS_TURBOMACHINERY_FROZEN) {
#       pragma omp parallel for if (n_cells > CS_THR_MIN)
        for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
          const cs_real_t romvom = -crom[c_id] * cell_f_vol[c_id];
          cs_rotation_add_coriolis_v(cs_glob_rotation,
                                     2*romvom, vela[c_id], trav_p[c_id]);
          if (irotce[c_id] > 0)
            cs_rotation_add_coriolis_v(cs_glob_rotation + irotce[c_id],
                                       romvom, vela[c_id], trav_p[c_id]);
        }
      }

      /* Reference frame rotation */
      else {
#       pragma omp parallel for if (n_cells > CS_THR_MIN)
        for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
          const cs_real_t romvom = -2 * crom[c_id] * cell_f_vol[c_id];
          cs_rotation_add_coriolis_v(cs_glob_rotation,
                                     romvom, vela[c_id], trav_p[c_id]);
        }
      }

    } /* iterns == 1 */

  }

  /* Implicit part;
     at the second call, fimp is not needed anymore */

  if (   iappel == 1
      && (   cs_glob_physical_constants->icorio == 1
          || iturbo == CS_TURBOMACHINERY_FROZEN)) {
    /* The theta-scheme for the Coriolis term is the same as the other terms */
    const cs_real_t thetap = eqp_u->theta;

    /* Reference frame + turbomachinery frozen rotors rotation */
    if (iturbo == CS_TURBOMACHINERY_FROZEN) {
#     pragma omp parallel for if (n_cells > CS_THR_MIN)
      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        const cs_real_t romvom = -crom[c_id] * cell_f_vol[c_id] * thetap;
        cs_rotation_add_coriolis_t(cs_glob_rotation, 2*romvom, fimp[c_id]);
        if (irotce[c_id] > 0) {
          cs_rotation_add_coriolis_t(cs_glob_rotation + irotce[c_id],
                                     romvom, fimp[c_id]);
        }
      }
    }

    /* Reference frame rotation */
    else {
#     pragma omp parallel for if (n_cells > CS_THR_MIN)
      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        const cs_real_t romvom = -2 * crom[c_id] * cell_f_vol[c_id] * thetap;
        cs_rotation_add_coriolis_t(cs_glob_rotation, romvom, fimp[c_id]);
      }
    }
  }

  /* Divergence of tensor Rij
     ------------------------
   * Non linear part of Rij for non-liear Eddy Viscosity Models */

  cs_real_3_t *cpro_divr = nullptr, *divt = nullptr;

  if (   iterns == 1
      && (   cs_glob_turb_model->itytur == 3
          || cs_glob_turb_model->iturb == CS_TURB_K_EPSILON_QUAD)) {

    cs_field_t *f_drij = cs_field_by_name_try("algo:divergence_rij");
    if (f_drij != nullptr) {
      assert(f_drij->dim == 3);
      cpro_divr
        = (cs_real_3_t *)f_drij->val;
    }
    else {
      CS_MALLOC_HD(divt, n_cells_ext, cs_real_3_t, cs_alloc_mode);
      cpro_divr = divt;
    }

    _div_rij(m,
             crom, brom,
             cpro_divr, c_st_vel,
             forbr, trava, trav);
  }

  /* Face diffusivity for the velocity
     --------------------------------- */

  _face_diff_vel(m, mq, eqp_u, viscf, viscb, viscfi, viscbi, viscce);

  CS_FREE_HD(viscce);

  /* Add Rusanov
     ----------- */

  if (cs_glob_turb_rans_model->irijnu == 2) {

    const cs_real_3_t *i_face_u_normal = mq->i_face_u_normal;
    const cs_real_3_t *b_face_u_normal = mq->b_face_u_normal;

    if (eqp_u->idften & CS_ISOTROPIC_DIFFUSION) {
      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
        viscf[f_id] = cs_math_fmax(viscf[f_id], 0.5*ipro_rusanov[f_id]);
      });
    }

    else if (eqp_u->idften & CS_ANISOTROPIC_LEFT_DIFFUSION) {
      ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
        const cs_real_t *n = i_face_u_normal[f_id];
        for (cs_lnum_t ii = 0; ii < 3; ii++) {
          for (cs_lnum_t jj = 0; jj < 3; jj++)
            viscf[9*f_id+3*jj+ii]
              = cs_math_fmax(viscf[9*f_id+3*jj+ii],
                             0.5*ipro_rusanov[f_id]*n[ii]*n[jj]);
        }
      });
    }

    const cs_real_t *bpro_rusanov = cs_field_by_name("b_rusanov_diff")->val;
    ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
      const cs_real_t *n = b_face_u_normal[f_id];
      for (cs_lnum_t ii = 0; ii < 3; ii++) {
        for (cs_lnum_t jj = 0; jj < 3; jj++)
          cofbfv[f_id][ii][jj] += bpro_rusanov[f_id]*n[ii]*n[jj];
      }
    });
  }

  /* External forces partially balanced with the pressure gradient
   * -----------------------------------------------------------------
   * (only for the first call, the second one is for error estimators) */

  if (iappel == 1 && vp_param->iphydr == 1)
    _ext_forces(m, mq, fp,
                ncepdc, icepdc,
                crom, croma, cromaa,
                gxyz, vela,
                tsexp, frcxt,
                cpro_divr, stf,
                ckupdc, dfrcxt);

  CS_FREE_HD(divt);
  CS_FREE_HD(icepdc);

  /* Solving of the 3x3xNcel coupled system
   ======================================== */

  cs_real_t *c_estim = nullptr;
  if (iappel == 1 && iespre != nullptr) {
    c_estim = iespre->val;
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      c_estim[c_id] = 0.;
    });
  }

  if (iappel == 2) {
    c_estim = cs_field_by_name_try("est_error_tot_2")->val;
    if (c_estim != nullptr) {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        c_estim[c_id] = 0.;
      });
    }
  }

  /* Use user source terms
     --------------------- */

  /* Explicit contribution due to implicit terms */

  if (iterns == 1) {
    cs_real_3_t *trav_p = (vp_param->nterup > 1) ? trava : trav;

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      for (cs_lnum_t ii = 0; ii < 3; ii++) {
        for (cs_lnum_t jj = 0; jj < 3; jj++)
          trav_p[c_id][ii] += tsimp[c_id][ii][jj] * vela[c_id][jj];
      }
    });
    ctx.wait();
  }

  /* Explicit user source terms are added */

  if (   vp_param->iphydr != 1
      || cs_glob_velocity_pressure_param->igpust != 1) {
    if (cs_glob_time_scheme->isno2t > 0) {
      if (iterns == 1)
        cs_axpy(n_cells*3, 1, (cs_real_t *)tsexp, (cs_real_t *)c_st_vel);
    }
    else
      cs_axpy(n_cells*3, 1, (cs_real_t *)tsexp, (cs_real_t *)trav);
  }

  CS_FREE_HD(loctsexp);

  /* Surface tension is added */

  if (vp_param->iphydr != 1 && cs_glob_vof_parameters->sigma_s > 0) {

    /* If source terms are time-extrapolated, they are stored in fields */
    if (cs_glob_time_scheme->isno2t > 0) {
      if (iterns == 1)
        cs_axpy(n_cells*3, 1, (cs_real_t *)stf, (cs_real_t *)c_st_vel);
    }
    else
      cs_axpy(n_cells*3, 1, (cs_real_t *)stf, (cs_real_t *)trav);
  }

  /* Implicit terms */

  if (iappel == 1) {
    if (cs_glob_time_scheme->isno2t > 0)
      cs_axpy(n_cells*3*3, -eqp_u->theta,
              (cs_real_t *)tsimp, (cs_real_t *)fimp);

    else {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        for (cs_lnum_t ii = 0; ii < 3; ii++) {
          for (cs_lnum_t jj = 0; jj < 3; jj++)
            fimp[c_id][ii][jj] += cs_math_fmax(-tsimp[c_id][ii][jj], 0.0);
        }
      });
      ctx.wait();
    }
  }

  CS_FREE_HD(loctsimp);

  /* Mass source terms
     ----------------- */

  if (eqp_u->n_volume_mass_injections > 0) {

    cs_lnum_t ncetsm = 0;
    int *itypsm = nullptr;
    const cs_lnum_t *icetsm = nullptr;
    cs_real_t *smacel_p = nullptr;
    cs_real_t *smacel_vel = nullptr;

    cs_volume_mass_injection_get_arrays(CS_F_(vel),
                                        &ncetsm,
                                        &icetsm,
                                        &itypsm,
                                        &smacel_vel,
                                        &smacel_p);

    cs_real_3_t *gavinj = nullptr;
    if (iterns == 1) {
      if (cs_glob_time_scheme->isno2t > 0)
        /* If source terms are extrapolated, stored in fields */
        gavinj = c_st_vel;
      else {
        if (vp_param->nterup == 1) /* If no inner iteration: in trav */
          gavinj = trav;
        else  /* Otherwise, in trava */
          gavinj = trava;
      }
    }

    cs_real_3_t *trav_p = (vp_param->nterup == 1) ? trav : trava;

    cs_mass_source_terms(iterns,
                         3,
                         ncetsm,
                         icetsm,
                         itypsm,
                         cell_f_vol,
                         (cs_real_t*)vela,
                         smacel_vel,
                         smacel_p,
                         (cs_real_t*)trav_p,
                         (cs_real_t*)fimp,
                         (cs_real_t*)gavinj);

  }

  BFT_FREE(stf);

  /* Right Hand Side initialization
     ------------------------------ */

  cs_real_3_t *smbr;
  CS_MALLOC_HD(smbr, n_cells_ext, cs_real_3_t, cs_alloc_mode);

  /* If source terms are extrapolated in time */
  if (cs_glob_time_scheme->isno2t > 0) {
    const cs_real_t thetp1 = 1.0 + thets;
    if (vp_param->nterup == 1) {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          smbr[c_id][ii] = trav[c_id][ii] + thetp1*c_st_vel[c_id][ii];
      });
    }
    else {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          smbr[c_id][ii] =   trav[c_id][ii] + trava[c_id][ii]
                           + thetp1*c_st_vel[c_id][ii];
      });
    }
  }

  /* No time extrapolation */
  else {
    if (vp_param->nterup == 1) {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          smbr[c_id][ii] = trav[c_id][ii];
      });
    }
    else {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          smbr[c_id][ii] = trav[c_id][ii] + trava[c_id][ii];
      });
    }
  }

  ctx.wait();

  /* Lagrangian: coupling feedback
     -----------------------------

     Order 2 on terms from the Lagrangian model would require to decompose
     the Lagrangian source terms into an implicit and explicit part, as
     is done for user source terms.

     For the time being, we do not try this.
  */

  if (   cs_glob_lagr_source_terms->ltsdyn == 1
      && cs_glob_lagr_time_scheme->iilagr == CS_LAGR_TWOWAY_COUPLING) {

    const cs_real_3_t *lagr_st_vel
      = (const cs_real_3_t *)cs_field_by_name_try("velocity_st_lagr")->val;

    cs_axpy(n_cells*3, 1, (const cs_real_t *)lagr_st_vel, (cs_real_t *)smbr);

    if (iappel == 1) {
      const cs_lnum_t itsli = cs_glob_lagr_source_terms->itsli;
      cs_real_t *st_val =   cs_glob_lagr_source_terms->st_val
                          + (itsli-1)*n_cells_ext;

#     pragma omp parallel for if (n_cells > CS_THR_MIN)
      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        cs_real_t st = cs_math_fmax(-st_val[c_id], 0.0);
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          fimp[c_id][ii][ii] += st;
      }
    }

  }

  /* Electric Arcs (Laplace Force) (No 2nd order in time yet)
     ----------------------------- */

  if (cs_glob_physical_model_flag[CS_ELECTRIC_ARCS] > 0) {
    const cs_real_3_t *lapla
      = (const cs_real_3_t *)cs_field_by_name("laplace_force")->val;

#   pragma omp parallel for if (n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        smbr[c_id][ii] += cell_f_vol[c_id] * lapla[c_id][ii];
    }
  }

  /* Solver parameters
     ----------------- */

  int icvflb = 0;
  if (cs_glob_physical_model_flag[CS_COMPRESSIBLE] > -1)
    icvflb = 1;

  cs_field_t *iestot = cs_field_by_name_try("est_error_tot_2");

  cs_real_3_t *eswork = nullptr;
  if (iespre != nullptr)
    CS_MALLOC_HD(eswork, n_cells_ext, cs_real_3_t, cs_alloc_mode);

  if (iappel == 1) {
    /* Store fimp as the velocity matrix is stored in codtiv call */
    cs_real_33_t *fimpcp = nullptr;
    CS_MALLOC_HD(fimpcp, n_cells_ext, cs_real_33_t, cs_alloc_mode);
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        for (cs_lnum_t jj = 0; jj < 3; jj++)
          fimpcp[c_id][ii][jj] = fimp[c_id][ii][jj];
    });

    int iescap = 0;
    if (iespre != nullptr)
      iescap = 1;

    cs_equation_param_t eqp_loc = *eqp_u;

    eqp_loc.istat  = -1;
    eqp_loc.idifft = -1;
    eqp_loc.iwgrec = 0;
    eqp_loc.blend_st = 0; //  Warning, may be overwritten if a field

    /* Warning: in case of convergence estimators, eswork gives the estimator
       of the predicted velocity */

    int *icvfli = cs_cf_boundary_conditions_get_icvfli();

    cs_equation_iterative_solve_vector(cs_glob_time_step_options->idtvar,
                                       iterns,
                                       CS_F_(vel)->id,
                                       nullptr,
                                       vp_model->ivisse,
                                       iescap,
                                       &eqp_loc,
                                       vela,
                                       velk,
                                       bc_coeffs_v,
                                       imasfl,
                                       bmasfl,
                                       viscfi,
                                       viscbi,
                                       viscf,
                                       viscb,
                                       secvif,
                                       secvib,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       icvflb,
                                       icvfli,
                                       fimp,
                                       smbr,
                                       vel,
                                       eswork);

    /* Compute kinetic energy balance for compressible algorithm
     * See H. Amino thesis */
    cs_thermal_model_kinetic_st_prepare(imasfl, bmasfl, vela, vel);

    /* Store inverse of the velocity matrix for the correction step
     * if needed (otherwise vitenp is used in cs_pressure_correction) */
    if (vp_param->rcfact == 1) {

      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        cs_real_t tensor[6] = {fimp[c_id][0][0]/crom[c_id],
                               fimp[c_id][1][1]/crom[c_id],
                               fimp[c_id][2][2]/crom[c_id],
                               fimp[c_id][1][0]/crom[c_id],
                               fimp[c_id][2][1]/crom[c_id],
                               fimp[c_id][2][0]/crom[c_id]};

        cs_math_sym_33_inv_cramer(tensor, da_uu[c_id]);

        for (cs_lnum_t ii = 0; ii < 6; ii++)
           da_uu[c_id][ii] *= cell_f_vol[c_id];
      });

      ctx.wait();

      cs_mesh_sync_var_sym_tens(da_uu);

    }

    /* Velocity-pression coupling: compute the vector T, stored in dttens,
     * cs_equation_iterative_solve_vector is called, only one sweep is done,
     * and dttens is initialized by 0, so that the advection/diffusion added
     * by cs_balance_vector is 0.
     *  nswrsp = -1 indicated that only one sweep is required and inc=0
     *  for boundary contitions on the weight matrix. */

    if (vp_param->ipucou == 1) {

      cs_real_3_t *vect;
      CS_MALLOC_HD(vect, n_cells_ext, cs_real_3_t, cs_alloc_mode);

      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        const int ind = has_disable_flag * c_id;
        const int c_act = (1 - (has_disable_flag * c_disable_flag[ind]));
        for (cs_lnum_t ii = 0; ii < 3; ii++) {
          smbr[c_id][ii] = c_act * cell_f_vol[c_id];
          vect[c_id][ii] = 0;
        }
      });

      iescap = 0;

      /* We do not take into account transpose of grad */
      int ivisep = 0;

      eqp_loc.nswrsm = -1;

      cs_equation_iterative_solve_vector(cs_glob_time_step_options->idtvar,
                                         iterns,
                                         CS_F_(vel)->id,
                                         nullptr,
                                         ivisep,
                                         iescap,
                                         &eqp_loc,
                                         vect,
                                         vect,
                                         bc_coeffs_v,
                                         imasfl,
                                         bmasfl,
                                         viscfi,
                                         viscbi,
                                         viscf,
                                         viscb,
                                         secvif,
                                         secvib,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         icvflb,
                                         nullptr,
                                         fimpcp,
                                         smbr,
                                         vect,
                                         nullptr);

      ctx.parallel_for(n_cells_ext, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        const cs_real_t rom = crom[c_id];
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          dttens[c_id][ii] = rom*vect[c_id][ii];
        for (cs_lnum_t ii = 3; ii < 6; ii++)
          dttens[c_id][ii] = 0;
      });

      CS_FREE_HD(vect);
    }

    /* The estimator on the predicted velocity is summed over the components */
    if (iespre != nullptr) {
      c_estim = iespre->val;
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          c_estim[c_id] += eswork[c_id][ii];
      });

    }

    CS_FREE_HD(fimpcp);
  }

  /* End of the construction of the total estimator:
   * RHS residual of (U^{n+1}, P^{n+1}) + rho*volume*(U^{n+1} - U^n)/dt */

  else if (iappel == 2) {

    /* No relaxation for steady case */
    int idtva0 = 0;
    int imasac = 0;

    cs_equation_param_t eqp_loc = *eqp_u;

    eqp_loc.istat  = -1;
    eqp_loc.idifft = -1;
    eqp_loc.iswdyn = -1;
    eqp_loc.nswrsm = -1;
    eqp_loc.iwgrec = 0;
    eqp_loc.blend_st = 0; /* Warning, may be overwritten if a field */
    eqp_loc.epsilo = -1;
    eqp_loc.epsrsm = -1;

    int *icvfli = cs_cf_boundary_conditions_get_icvfli();

    cs_balance_vector(idtva0,
                      CS_F_(vel)->id,
                      imasac,
                      1,
                      vp_model->ivisse,
                      &eqp_loc,
                      vel,
                      vel,
                      bc_coeffs_v,
                      imasfl,
                      bmasfl,
                      viscf,
                      viscb,
                      secvif,
                      secvib,
                      nullptr,
                      nullptr,
                      nullptr,
                      icvflb,
                      icvfli,
                      nullptr,
                      nullptr,
                      smbr);

    c_estim = iestot->val;

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      c_estim[c_id] = 0;
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        c_estim[c_id] += cs_math_pow2(smbr[c_id][ii] / cell_f_vol[c_id]);
    });
  }

  CS_FREE_HD(fimp);
  CS_FREE_HD(smbr);
  CS_FREE_HD(eswork);

  /* Finalaze estimators + logging */

  f = cs_field_by_name_try("algo:predicted_velocity");
  if (f != nullptr) {
    cs_real_3_t *pre_vel = (cs_real_3_t *)f->val;

    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      for (cs_lnum_t ii = 0; ii < 3; ii++) {
        pre_vel[c_id][ii] = vel[c_id][ii];
      }
    });
  }

  if (iappel == 1) {

    /* Estimator on the predicted velocity:
     * square root (norm) or square root of the sum times the volume (L2 norm) */
    if (iespre != nullptr) {
      c_estim = iespre->val;
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        c_estim[c_id] = sqrt(c_estim[c_id] * cell_f_vol[c_id]);
      });
      ctx.wait();
    }

    /* Norm logging */
    if (eqp_u->verbosity > 1) {

      cs_real_t rnormx = -1.0, rnormn = HUGE_VAL;
      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        cs_real_t vitnor = cs_math_3_norm(vel[c_id]);
        rnormx = cs_math_fmax(rnormx, vitnor);
        rnormn = cs_math_fmin(rnormn, vitnor);
      }

      cs_parall_max(1, CS_REAL_TYPE, &rnormx);
      cs_parall_min(1, CS_REAL_TYPE, &rnormn);

      bft_printf(_("Maximum velocity after prediction %10.12e\n"
                   "Minimum velocity after prediction %10.12e\n"),
                 rnormx, rnormn);
    }

  }

  /* Estimator on the whole Navier-Stokes:
   * square root (norm) or square root of the sum times the volume (L2 norm) */
  else if (iappel == 2) {
    if (iestot != nullptr) {
      c_estim = iestot->val;

      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        c_estim[c_id] = sqrt(c_estim[c_id]*cell_f_vol[c_id]);
      });
      ctx.wait();
    }
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute a hydrostatic pressure \f$ P_{hydro} \f$ solving an
 *        a priori simplified momentum equation:
 *
 * \param[out]    grdphd         the a priori hydrostatic pressure gradient
 *                              \f$ \partial _x (P_{hydro}) \f$
 * \param[in]     iterns        gravity
 * \param[in]     iterns        Navier-Stokes iteration number
 */
/*----------------------------------------------------------------------------*/

static void
_hydrostatic_pressure_prediction(cs_real_t        grdphd[][3],
                                 const cs_real_t  gxyz[3],
                                 int              iterns)
{
  const cs_mesh_t *m = cs_glob_mesh;
  const cs_mesh_quantities_t *mq = cs_glob_mesh_quantities;

  const cs_lnum_t n_cells = m->n_cells;
  const cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;
  const cs_lnum_t n_i_faces = m->n_i_faces;
  const cs_lnum_t n_b_faces = m->n_b_faces;
  const int idtvar = cs_glob_time_step_options->idtvar;

  const cs_lnum_t *b_face_cells = m->b_face_cells;

  cs_real_t *prhyd = cs_field_by_name("hydrostatic_pressure_prd")->val;

  const cs_real_t *crom = CS_F_(rho)->val;

  const int kimasf = cs_field_key_id("inner_mass_flux_id");
  const int kbmasf = cs_field_key_id("boundary_mass_flux_id");
  const int iflmas = cs_field_get_key_int(CS_F_(vel), kimasf);
  const int iflmab = cs_field_get_key_int(CS_F_(vel), kbmasf);

  cs_real_t *imasfl = cs_field_by_id(iflmas)->val;
  cs_real_t *bmasfl = cs_field_by_id(iflmab)->val;

  cs_dispatch_context ctx;

  /* Boundary conditions for delta P */
  cs_field_bc_coeffs_t bc_coeffs_dp;
  cs_field_bc_coeffs_init(&bc_coeffs_dp);
  CS_MALLOC_HD(bc_coeffs_dp.a,  n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs_dp.af, n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs_dp.b,  n_b_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(bc_coeffs_dp.bf, n_b_faces, cs_real_t, cs_alloc_mode);

  cs_real_t *coefap = bc_coeffs_dp.a;
  cs_real_t *cofafp = bc_coeffs_dp.af;
  cs_real_t *coefbp = bc_coeffs_dp.b;
  cs_real_t *cofbfp = bc_coeffs_dp.bf;

  /*
   * Solve a diffusion equation with source term to obtain
   * the a priori hydrostatic pressure
   * ----------------------------------------------------- */

  cs_real_t *xinvro, *rovsdt, *rhs;
  CS_MALLOC_HD(xinvro, n_cells_ext, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(rovsdt, n_cells_ext, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(rhs, n_cells_ext, cs_real_t, cs_alloc_mode);

  /* Initialization of the variable to solve from the interior cells */

  ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
    xinvro[c_id] = 1. / crom[c_id];
    rovsdt[c_id] = 0;
    rhs[c_id] = 0;
  });

  /* Allocate work arrays */
  cs_real_t *viscf, *viscb;
  CS_MALLOC_HD(viscf, n_i_faces, cs_real_t, cs_alloc_mode);
  CS_MALLOC_HD(viscb, n_b_faces, cs_real_t, cs_alloc_mode);

  /* Viscosity (k_t := 1/rho ) */

  cs_face_viscosity(m, mq,
                    1, /* harmonic mean */
                    xinvro,
                    viscf, viscb);

  /* Neumann boundary condition for the pressure increment */

  const cs_real_t *distb = mq->b_dist;
  const cs_real_3_t *b_face_u_normal = (const cs_real_3_t *)mq->b_face_u_normal;

  ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t f_id) {
    const cs_lnum_t c_id = b_face_cells[f_id];

    /* Prescribe the pressure gradient: kt.grd(Phyd)|_b = (g.n)|_b */

    cs_real_t hint = 1. / (crom[c_id] * distb[f_id]);
    cs_real_t qimp = - cs_math_3_dot_product(b_face_u_normal[f_id],
                                             gxyz);

    /* Neumann for scalar dp */

    // Gradient BCs
    coefap[f_id] = -qimp/cs_math_fmax(hint, 1.e-300);
    coefbp[f_id] = 1.;

    // Flux BCs
    cofafp[f_id] = qimp;
    cofbfp[f_id] = 0.;

  });

  /* Solve the diffusion equation.

     By default, the hydrostatic pressure variable is resolved with 5 sweeps
     for the reconstruction gradient. Here we make the assumption that the
     mesh is orthogonal (any reconstruction gradient is done for the
     hydrostatic pressure variable). */

  const cs_equation_param_t *eqp_p
    = cs_field_get_equation_param_const(CS_F_(p));

  cs_equation_param_t eqp_loc = *eqp_p;

  eqp_loc.iconv = 0;
  eqp_loc.istat = 0;
  eqp_loc.icoupl = -1;
  eqp_loc.ndircl = 0;
  eqp_loc.idiff  = 1;
  eqp_loc.idifft = -1;
  eqp_loc.idften = CS_ISOTROPIC_DIFFUSION;
  eqp_loc.nswrsm = 1;    /* no reconstruction gradient
                            (important for mesh with reconstruction) */
  eqp_loc.iwgrec = 0;    /* Warning, may be overwritten if a field */
  eqp_loc.blend_st = 0;  /* Warning, may be overwritten if a field */

  cs_real_t *dpvar;
  CS_MALLOC_HD(dpvar, n_cells_ext, cs_real_t, cs_alloc_mode);

  const char var_name[] = "Prhydro";

  cs_equation_iterative_solve_scalar(idtvar,
                                     iterns,
                                     -1,     /* field id */
                                     var_name,
                                     0,      /* iescap */
                                     0,      /* imucpp */
                                     -1,     /* normp */
                                     &eqp_loc,
                                     prhyd, prhyd,
                                     &bc_coeffs_dp,
                                     imasfl, bmasfl,
                                     viscf, viscb,
                                     viscf, viscb,
                                     nullptr,   /* viscel */
                                     nullptr,   /* weighf */
                                     nullptr,   /* weighb */
                                     0,      /* icvflb (upwind conv. flux) */
                                     nullptr,   /* icvfli */
                                     rovsdt,
                                     rhs,
                                     prhyd, dpvar,
                                     nullptr,   /* xcpp */
                                     nullptr);  /* eswork */

  CS_FREE_HD(dpvar);

  cs_halo_type_t halo_type = CS_HALO_STANDARD;
  cs_gradient_type_t gradient_type = CS_GRADIENT_GREEN_ITER;

  cs_gradient_type_by_imrgra(eqp_loc.imrgra,
                             &gradient_type,
                             &halo_type);

  cs_gradient_scalar(var_name,
                     gradient_type,
                     halo_type,
                     1, /* inc */
                     1, /* n_r_sweeps */
                     0, /* hyd_p_flag */
                     1,             /* w_stride */
                     eqp_loc.verbosity,
                     static_cast<cs_gradient_limit_t>(eqp_loc.imligr),
                     eqp_loc.epsrgr,
                     eqp_loc.climgr,
                     nullptr, /* f_ext */
                     &bc_coeffs_dp,
                     prhyd,
                     xinvro,
                     nullptr,
                     grdphd);

  /* Free memory */

  CS_FREE_HD(viscf);
  CS_FREE_HD(viscb);

  CS_FREE_HD(xinvro);
  CS_FREE_HD(rovsdt);
  CS_FREE_HD(rhs);

  CS_FREE_HD(bc_coeffs_dp.a);
  CS_FREE_HD(bc_coeffs_dp.af);
  CS_FREE_HD(bc_coeffs_dp.b);
  CS_FREE_HD(bc_coeffs_dp.bf);
}

/*============================================================================
 * Fortran wrapper function definitions
 *============================================================================*/

void
cs_f_navier_stokes_total_pressure(void)
{
  const cs_fluid_properties_t *fp = cs_glob_fluid_properties;
  const cs_real_t *gxyz = cs_glob_physical_constants->gravity;
  const cs_real_t *xyzp0 = fp->xyzp0;
#if defined(HAVE_ACCEL)
  cs_real_t *_gxyz = nullptr, *_xyzp0 = nullptr;
  if (cs_get_device_id() > -1) {
    CS_MALLOC_HD(_gxyz, 3, cs_real_t, cs_alloc_mode);
    CS_MALLOC_HD(_xyzp0, 3, cs_real_t, cs_alloc_mode);
    for (int i = 0; i < 3; i++) {
      _gxyz[i] = cs_glob_physical_constants->gravity[i];
      _xyzp0[i] = fp->xyzp0[i];
    }

    cs_mem_advise_set_read_mostly(_gxyz);
    cs_mem_advise_set_read_mostly(_xyzp0);

    xyzp0 = _xyzp0;
    gxyz = _gxyz;
  }
#endif

  cs_solve_navier_stokes_update_total_pressure(cs_glob_mesh,
                                               cs_glob_mesh_quantities,
                                               cs_glob_fluid_properties,
                                               gxyz,
                                               xyzp0);

#if defined(HAVE_ACCEL)
  CS_FREE_HD(_gxyz);
  CS_FREE_HD(_xyzp0);
#endif

}

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Update total pressure (defined as a post-processed property).
 *
 * For the compressible module, the solved pressure is already
 * the total pressure.
 *
 * Note: for Eddy Viscosity Models, the TKE may be included in the
 * solved pressure.
 *
 * \param[in]     m      pointer to mesh structure
 * \param[in]     mq     pointer to mesh quantities structure
 * \param[in]     fp     pointer to fluid properties structure
 * \param[in]     gxyz   gravity
 * \param[in]     xyzp0  indicator for filling of reference point for
 *                       total pressure
 */
/*----------------------------------------------------------------------------*/

void
cs_solve_navier_stokes_update_total_pressure
  (const cs_mesh_t              *m,
   const cs_mesh_quantities_t   *mq,
   const cs_fluid_properties_t  *fp,
   const cs_real_t               gxyz[3],
   const cs_real_t               xyzp0[3])
{
  /* TODO: use a function pointer here to adapt to different cases */

  cs_field_t *f = cs_field_by_name_try("total_pressure");

  if ((CS_F_(p) == nullptr) || (f == nullptr))
    return;

  const cs_lnum_t n_cells = m->n_cells;

  const cs_real_3_t *cell_cen = (const cs_real_3_t *)mq->cell_cen;
  const cs_real_t p0 = fp->p0, pred0 = fp->pred0, ro0 = fp->ro0;

  cs_real_t *cpro_prtot = f->val;
  const cs_real_t *cvar_pr = CS_F_(p)->val;

  const cs_field_t *f_k = CS_F_(k);
  const cs_real_t *cvar_k = (f_k != nullptr) ? CS_F_(k)->val : nullptr;
  const cs_real_t *cpro_rho = CS_F_(rho)->val;

  const cs_real_3_t *cpro_momst = nullptr;

  if (cs_glob_atmo_option->open_bcs_treatment != 0)
    cpro_momst
      = (const cs_real_3_t *)cs_field_by_name("momentum_source_terms")->val;

  cs_dispatch_context ctx;

  /* Update cell values */

  bool is_eddy_model
    =  (  (   cs_glob_turb_model->itytur == 2
           || cs_glob_turb_model->itytur == 5
           || cs_glob_turb_model->iturb == CS_TURB_K_OMEGA)
        && cs_glob_turb_rans_model->igrhok != 1);


  if (cpro_momst == nullptr) {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      cpro_prtot[c_id] =   cvar_pr[c_id]
                         + ro0 * cs_math_3_distance_dot_product(xyzp0,
                                                                cell_cen[c_id],
                                                                gxyz)
                         + p0 - pred0;

      /* For Eddy Viscosity Models, "2/3 rho k"
         is included in the solved pressure */
      if (is_eddy_model)
        cpro_prtot[c_id] -= 2.0/3 * cpro_rho[c_id]*cvar_k[c_id];
    });
  }
  else {
    ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      cpro_prtot[c_id] =   cvar_pr[c_id]
                         + ro0 * cs_math_3_distance_dot_product(xyzp0,
                                                                cell_cen[c_id],
                                                                gxyz)
                         + p0 - pred0
                         - cs_math_3_distance_dot_product(xyzp0,
                                                          cell_cen[c_id],
                                                          cpro_momst[c_id]);
      /* For Eddy Viscosity Models, "2/3 rho k"
         is included in the solved pressure */
      if (is_eddy_model)
        cpro_prtot[c_id] -= 2.0/3 * cpro_rho[c_id]*cvar_k[c_id];
    });
  }

}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Solve Navier-Stokes equations for incompressible or slightly
 *        compressible flows for one time step. Both convection-diffusion
 *        and continuity steps are performed.
 *
 * \param[in]     iterns        index of the iteration on Navier-Stokes
 * \param[in]     icvrge        convergence indicator
 * \param[in]     itrale        number of the current ALE iteration
 * \param[in]     isostd        indicator of standard outlet
 *                              + index of the reference face
 * \param[in]     ckupdc        head loss coefficients, if present
 */
/*----------------------------------------------------------------------------*/

void
cs_solve_navier_stokes(const int        iterns,
                       int             *icvrge,
                       const int        itrale,
                       const int        isostd[],
                       const cs_real_t  ckupdc[][6])

{
  cs_mesh_t *m = cs_glob_mesh;
  cs_mesh_quantities_t *mq = cs_glob_mesh_quantities;

  cs_lnum_t n_cells = m->n_cells;
  cs_lnum_t n_cells_ext = m->n_cells_with_ghosts;
  cs_lnum_t n_i_faces = m->n_i_faces;
  cs_lnum_t n_b_faces = m->n_b_faces;

  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *restrict)m->b_face_cells;

  const cs_time_step_t *ts = cs_glob_time_step;
  const cs_wall_condensation_t *w_condensation = cs_glob_wall_condensation;
  const cs_vof_parameters_t *vof_param = cs_glob_vof_parameters;
  const cs_fluid_properties_t *fluid_props = cs_glob_fluid_properties;
  const cs_velocity_pressure_model_t *vp_model = cs_glob_velocity_pressure_model;
  cs_velocity_pressure_param_t *vp_param = cs_get_glob_velocity_pressure_param();
  int has_disable_flag = mq->has_disable_flag;
  int *c_disable_flag = mq->c_disable_flag;

  const cs_equation_param_t *eqp_p
    = cs_field_get_equation_param_const(CS_F_(p));

  const cs_equation_param_t *eqp_u
    = cs_field_get_equation_param_const(CS_F_(vel));

  int nbrcpl = cs_sat_coupling_n_couplings();

  cs_dispatch_context ctx, ctx_c;
#if defined(HAVE_CUDA)
  ctx_c.set_cuda_stream(cs_cuda_get_stream(1));
#endif

  const cs_real_t *xyzp0 = fluid_props->xyzp0;
  const cs_real_t *gxyz = cs_glob_physical_constants->gravity;
#if defined(HAVE_ACCEL)
  cs_real_t *_gxyz = nullptr, *_xyzp0 = nullptr;
  if (cs_get_device_id() > -1) {
    CS_MALLOC_HD(_gxyz, 3, cs_real_t, cs_alloc_mode);
    CS_MALLOC_HD(_xyzp0, 3, cs_real_t, cs_alloc_mode);
    for (int i = 0; i < 3; i++) {
      _gxyz[i] = cs_glob_physical_constants->gravity[i];
      _xyzp0[i] = fluid_props->xyzp0[i];
    }

    cs_mem_advise_set_read_mostly(_gxyz);
    cs_mem_advise_set_read_mostly(_xyzp0);

    xyzp0 = _xyzp0;
    gxyz = _gxyz;
  }
#endif

  /* Initialization
   * -------------- */

  cs_real_t *dt = CS_F_(dt)->val;
  cs_real_t *cvar_pr = CS_F_(p)->val;
  cs_real_3_t *vel = (cs_real_3_t *)CS_F_(vel)->val;
  cs_real_3_t *vela = (cs_real_3_t *)CS_F_(vel)->val_pre;

  /* Map some specific field arrays */
  cs_field_t *f_dttens = cs_field_by_name_try("dttens");
  cs_real_6_t *dttens = nullptr;
  if (f_dttens != nullptr)
    dttens = (cs_real_6_t *)f_dttens->val;

  /* Pointer to velocity at sub iteration k for velocity-pressure
     inner iterations */
  cs_real_3_t *uvwk = nullptr, *velk = nullptr;

  if (vp_param->nterup > 1) {

    const cs_real_t *cell_f_vol = mq->cell_f_vol;

    CS_MALLOC_HD(uvwk, n_cells_ext, cs_real_3_t, cs_alloc_mode);

    cs_array_copy<cs_real_t>(3*n_cells,
                             (const cs_real_t *)vel,
                             (cs_real_t *)uvwk);

    /* Compute the L2 velocity norm
       (it is zero at the first time step, so we recompute it) */
    if (iterns == 1 || fabs(vp_param->xnrmu0) <= 0) {
      cs_real_t xnrtmp = 0.0;
#     pragma omp parallel for reduction(+:xnrtmp) if(n_cells > CS_THR_MIN)
      for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
        xnrtmp += cs_math_3_dot_product(vel[c_id],
                                        vel[c_id])*cell_f_vol[c_id];
      }
      cs_parall_sum(1, CS_REAL_TYPE, &xnrtmp);
      vp_param->xnrmu0 = xnrtmp;

      /* When coupling between multiple instances of code_saturne,
         we compute the total velocity norm.
         This is required so that one instance does not stop earlier than
         the others (the numerical options should still be checked) */
      cs_real_t xnrdis[1] = {0}, xnr_mu[1] = {vp_param->xnrmu0};
      for (int cpl_id = 0; cpl_id < nbrcpl; cpl_id++) {
        cs_sat_coupling_array_exchange(cpl_id,
                                       1, /* nbrdis */
                                       1, /* nbrloc */
                                       xnr_mu,
                                       xnrdis);
        xnr_mu[0] += xnrdis[0];
      }
      vp_param->xnrmu0 = sqrt(xnr_mu[0]);
    }

    /* Handle parallelism or periodicity of uvwk and pressure */
    ctx.wait();
    cs_mesh_sync_var_scal(cvar_pr);
    cs_mesh_sync_var_vect((cs_real_t *)uvwk);
    velk = uvwk;

  }
  else
    velk = vela;

  /* Physical quantities */
  const cs_real_t *viscl = CS_F_(mu)->val;
  const cs_real_t *visct = CS_F_(mu_t)->val;

  /* Pointers to properties */
  cs_real_t *crom_eos = CS_F_(rho)->val;
  const cs_real_t *brom_eos = CS_F_(rho_b)->val;
  const cs_real_t *croma = nullptr, *broma = nullptr;

  const cs_real_t *brom = nullptr;
  cs_real_t *crom, *cpro_rho_mass = nullptr, *bpro_rho_mass = nullptr;

  const cs_real_t *cromk1 = nullptr;
  cs_real_t *cpro_rho_k1 = nullptr;
  cs_real_t *cpro_rho_tc = nullptr, *bpro_rho_tc = nullptr;
  cs_real_t theta = eqp_u->theta;

  if (   fluid_props->irovar == 1
      && (   vp_model->idilat > 1
          || vof_param->vof_model > 0
          || cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 3)) {

    /* If iterns = 1: this is density at time n */
    cpro_rho_mass = cs_field_by_name("density_mass")->val;
    bpro_rho_mass = cs_field_by_name("boundary_density_mass")->val;

    /* Time interpolated density */
    if (theta < 1.0 && vp_param->itpcol == 0) {
      croma = CS_F_(rho)->val_pre;
      broma = CS_F_(rho_b)->val_pre;
      CS_MALLOC_HD(bpro_rho_tc, n_b_faces, cs_real_t, cs_alloc_mode);
      CS_MALLOC_HD(cpro_rho_tc, n_cells_ext, cs_real_t, cs_alloc_mode);

      ctx.parallel_for(n_cells_ext, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        cpro_rho_tc[c_id] =   theta * cpro_rho_mass[c_id]
                            + (1.0 - theta) * croma[c_id];
      });

      ctx.parallel_for(n_b_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
        bpro_rho_tc[face_id] =   theta * bpro_rho_mass[face_id]
                               + (1.0 - theta) * broma[face_id];
      });

      ctx.wait();

      crom = cpro_rho_tc;
      cromk1 = cpro_rho_tc;  /* rho at time n+1/2,k-1 */
      brom = bpro_rho_tc;
    }
    else {
      CS_MALLOC_HD(cpro_rho_k1, n_cells_ext, cs_real_t, cs_alloc_mode);
      cs_array_copy<cs_real_t>(n_cells_ext, cpro_rho_mass, cpro_rho_k1);

      ctx.wait();

      crom = cpro_rho_mass;
      cromk1 = cpro_rho_k1;  /* rho at time n+1/2,k-1 */
      brom = bpro_rho_mass;
    }
  }

  /* Weakly variable density algo. (idilat <=1) or constant density */
  else {
    crom = crom_eos;
    cromk1 = crom_eos;   /* rho at time n+1/2,k-1 */
    brom = brom_eos;
  }

  /* Prediction of the mass flux in case of Low Mach compressible algorithm
     ---------------------------------------------------------------------- */

  if (   (vp_model->idilat == 2 || vp_model->idilat == 3)
      && ts->nt_cur > 1
      && vp_param->ipredfl != 0)
    _cs_mass_flux_prediction(m, mq, dt);

  /* Hydrostatic pressure prediction in case of Low Mach compressible algorithm
     ---------------------------------------------------------------------------*/

  cs_real_3_t *grdphd = nullptr;
  if (vp_param->iphydr == 2) {
    CS_MALLOC_HD(grdphd, n_cells_ext, cs_real_3_t, cs_alloc_mode);
    _hydrostatic_pressure_prediction(grdphd, gxyz, iterns);
  }

  /* Pressure resolution and computation of mass flux for compressible flow
     ---------------------------------------------------------------------- */

  /* Note, for the compressible algorithm written in pressure increment,
   * this step is merged with the pressure correction step of the incompressible
   * algorithm */

  if (   cs_glob_physical_model_flag[CS_COMPRESSIBLE] > -1
      && cs_glob_physical_model_flag[CS_COMPRESSIBLE] != 3) {
    if (eqp_p->verbosity >= 1)
      bft_printf("** SOLVING MASS BALANCE EQUATION\n");

    cs_cf_convective_mass_flux(iterns);
  }

  /* VoF: compute liquid-vapor mass transfer term (cavitating flows)
     --------------------------------------------------------------- */

  if (vof_param->vof_model & CS_VOF_MERKLE_MASS_TRANSFER) {
    const cs_real_t *cpro_prtot = cs_field_by_name("total_pressure")->val;
    const cs_real_t *cvara_voidf = cs_field_by_name("void_fraction")->val_pre;
    cs_cavitation_compute_source_term(cpro_prtot, cvara_voidf);
  }

  /* Velocity prediction step
     ------------------------ */

  bool irijnu_1 = false;
  if (   cs_glob_turb_model->itytur == 3
      && cs_glob_turb_rans_model->irijnu == 1)
    irijnu_1 = true;

  if (eqp_u->verbosity > 0)
    bft_printf("** SOLVING VELOCITY\n");

  cs_real_t *viscf = nullptr, *viscb = nullptr;
  cs_real_t *secvib = nullptr, *secvif = nullptr;
  cs_real_t *viscfi = nullptr, *viscbi = nullptr;
  cs_real_t *wvisbi = nullptr, *wvisfi = nullptr;
  cs_real_3_t *frcxt = nullptr;

  static cs_real_3_t *trava = nullptr;  /* TODO: pass this as argument to calling
                                        function when that is moved to C,
                                        so as to avoid requiring a static
                                        variable. */

  if (vp_param->nterup > 1 && trava == nullptr)
    CS_MALLOC_HD(trava, n_cells_ext, cs_real_3_t, cs_alloc_mode);

  if (vp_model->ivisse == 1) {
    CS_MALLOC_HD(secvif, n_i_faces, cs_real_t, cs_alloc_mode);
    CS_MALLOC_HD(secvib, n_b_faces, cs_real_t, cs_alloc_mode);
  }

  if (eqp_u->idften & CS_ISOTROPIC_DIFFUSION) {
    CS_MALLOC_HD(viscf, n_i_faces, cs_real_t, cs_alloc_mode);
    CS_MALLOC_HD(viscb, n_b_faces, cs_real_t, cs_alloc_mode);
    if (irijnu_1) {
      CS_MALLOC_HD(wvisfi, n_i_faces, cs_real_t, cs_alloc_mode);
      CS_MALLOC_HD(wvisbi, n_b_faces, cs_real_t, cs_alloc_mode);
      viscfi = wvisfi;
      viscbi = wvisbi;
    }
    else {
      viscfi = viscf;
      viscbi = viscb;
    }
  }
  else if (eqp_u->idften & CS_ANISOTROPIC_LEFT_DIFFUSION) {
    CS_MALLOC_HD(viscb, n_b_faces, cs_real_t, cs_alloc_mode);
    CS_MALLOC_HD(viscf, 9*n_i_faces, cs_real_t, cs_alloc_mode);
    if (irijnu_1) {
      CS_MALLOC_HD(wvisbi, n_b_faces, cs_real_t, cs_alloc_mode);
      CS_MALLOC_HD(wvisfi, 9*n_i_faces, cs_real_t, cs_alloc_mode);
      viscfi = wvisfi;
      viscbi = wvisbi;
    }
    else {
      viscfi = viscf;
      viscbi = viscb;
    }
  }

  cs_real_3_t *trav = nullptr, *dfrcxt = nullptr;
  cs_real_6_t *da_uu = nullptr;

  CS_MALLOC_HD(trav, n_cells_ext, cs_real_3_t, cs_alloc_mode);
  CS_MALLOC_HD(da_uu, n_cells_ext, cs_real_6_t, cs_alloc_mode);
  CS_MALLOC_HD(dfrcxt, n_cells_ext, cs_real_3_t, cs_alloc_mode);

  if (vp_param->iphydr == 1)
    frcxt = (cs_real_3_t *)cs_field_by_name("volume_forces")->val;

  /* Pointers to BC coefficients */
  cs_field_bc_coeffs_t *bc_coeffs_vel = CS_F_(vel)->bc_coeffs;
  cs_real_3_t *coefau = (cs_real_3_t *)CS_F_(vel)->bc_coeffs->a;
  cs_real_3_t *cofafu = (cs_real_3_t *)CS_F_(vel)->bc_coeffs->af;

  if (vp_param->staggered == 0)
    _velocity_prediction(m,
                         mq,
                         1,
                         iterns,
                         dt,
                         vel,
                         vela,
                         velk,
                         da_uu,
                         bc_coeffs_vel,
                         ckupdc,
                         frcxt,
                         grdphd,
                         gxyz,
                         xyzp0,
                         trava,
                         dfrcxt,
                         dttens,
                         trav,
                         viscf,
                         viscb,
                         viscfi,
                         viscbi,
                         secvif,
                         secvib);
  else {
    /* Account for external forces partially balanced by the pressure gradient
       (only for the first call; the second one is for error estimators) */
    if (vp_param->iphydr == 1) {
      const cs_real_t ro0 = fluid_props->ro0;

      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        //const int is_active = cs_mesh_quantities_cell_is_active(mq, c_id);
        const int ind = has_disable_flag * c_id;
        const int is_active = (1 - (has_disable_flag * c_disable_flag[ind]));
        const cs_real_t drom =  (crom[c_id] - ro0) * is_active;
        for (cs_lnum_t ii = 0; ii < 3; ii++)
          dfrcxt[c_id][ii] = drom * gxyz[ii] - frcxt[c_id][ii]*is_active;
      });

      ctx.wait(); // needed for the following synchronization

      cs_mesh_sync_var_vect((cs_real_t *)dfrcxt);
    }
  }

  /* Bad cells regularisation */
  cs_bad_cells_regularisation_vector(vel, 1);

  /* Exit if no pressure-continuity:
   * update mass fluxes and return */

  const int kimasf = cs_field_key_id("inner_mass_flux_id");
  const int kbmasf = cs_field_key_id("boundary_mass_flux_id");
  const int iflmas = cs_field_get_key_int(CS_F_(vel), kimasf);
  const int iflmab = cs_field_get_key_int(CS_F_(vel), kbmasf);

  cs_real_t *imasfl = cs_field_by_id(iflmas)->val;
  cs_real_t *bmasfl = cs_field_by_id(iflmab)->val;

  if (vp_param->iprco < 1) {
    int iflmb0 = 1;
    if (cs_glob_ale > CS_ALE_NONE)
      iflmb0 = 0;

    cs_mass_flux(m,
                 mq,
                 CS_F_(vel)->id,
                 1,  /* itypfl */
                 iflmb0,
                 1,  /* init */
                 1,  /* inc */
                 eqp_u->imrgra,
                 eqp_u->nswrgr,
                 static_cast<cs_gradient_limit_t>(eqp_u->imligr),
                 eqp_u->verbosity,
                 eqp_u->epsrgr,
                 eqp_u->climgr,
                 crom, brom,
                 vel,
                 bc_coeffs_vel,
                 imasfl, bmasfl);

    /* In the ALE framework, we add the mesh velocity */

    if (cs_glob_ale > CS_ALE_NONE)
      _mesh_velocity_mass_flux(m, mq,
                               dt,
                               crom, brom,
                               imasfl, bmasfl);

    /* Ajout de la vitesse du solide dans le flux convectif,
     * si le maillage est mobile (solide rigide)
     * En turbomachine, on connait exactement la vitesse de maillage a ajouter */

    if (cs_turbomachinery_get_model() > CS_TURBOMACHINERY_NONE)
      _turbomachinery_mass_flux(m,
                                mq,
                                crom, brom,
                                imasfl, bmasfl);

    CS_FREE_HD(trav);
    CS_FREE_HD(da_uu);
    CS_FREE_HD(dfrcxt);

    CS_FREE_HD(viscb);
    CS_FREE_HD(viscf);

    CS_FREE_HD(secvib);
    CS_FREE_HD(secvif);

    CS_FREE_HD(grdphd);

    CS_FREE_HD(cpro_rho_tc);
    CS_FREE_HD(bpro_rho_tc);

    CS_FREE_HD(wvisfi);
    CS_FREE_HD(wvisbi);

    CS_FREE_HD(uvwk);

    CS_FREE_HD(viscb);
    CS_FREE_HD(viscf);

#if defined(HAVE_ACCEL)
    CS_FREE_HD(_gxyz);
    CS_FREE_HD(_xyzp0);
#endif

    return;
  }

  /* Update mesh for unsteady turbomachinery computations */

  cs_real_t rs_ell[2] = {0, 0};

  if (   iterns == 1
      && cs_turbomachinery_get_model() == CS_TURBOMACHINERY_TRANSIENT) {

    cs_turbomachinery_update_mesh(rs_ell);

    const cs_real_t t1 = cs_timer_wtime();

    m = cs_glob_mesh;
    mq = cs_glob_mesh_quantities;
    ts = cs_glob_time_step;

    n_cells = m->n_cells;
    n_cells_ext = m->n_cells_with_ghosts;
    n_i_faces = m->n_i_faces;
    n_b_faces = m->n_b_faces;

    b_face_cells = (const cs_lnum_t *restrict)m->b_face_cells;

    if (cs_turbomachinery_get_n_couplings() < 1) {

      for (cs_lnum_t face_id = 0; face_id < n_b_faces; face_id++) {
        /* Cancel the mass flux for symmetry BC */
        if (cs_glob_bc_type[face_id] == CS_SYMMETRY)
          mq->b_sym_flag[face_id] = 0;
        else
          mq->b_sym_flag[face_id] = 1;
      }

      /* Resize temporary internal faces arrays */

      CS_FREE_HD(viscf);
      if (eqp_u->idften & CS_ISOTROPIC_DIFFUSION)
        CS_MALLOC_HD(viscf, n_i_faces, cs_real_t, cs_alloc_mode);
      else if (eqp_u->idften & CS_ANISOTROPIC_LEFT_DIFFUSION)
        CS_MALLOC_HD(viscf, 9*n_i_faces, cs_real_t, cs_alloc_mode);

      if (wvisfi != nullptr) {
        CS_FREE_HD(viscfi);
        if (eqp_u->idften == 1) {
          if (irijnu_1) {
            CS_MALLOC_HD(wvisfi, n_i_faces, cs_real_t, cs_alloc_mode);
            viscfi = wvisfi;
          }
          else
            viscfi = viscf;
        }
        else if (eqp_u->idften == 6) {
          if (irijnu_1) {
            CS_MALLOC_HD(wvisfi, 9*n_i_faces, cs_real_t, cs_alloc_mode);
            viscfi = wvisfi;
          }
          else
            viscfi = viscf;
        }
      }

      if (secvif != nullptr) {
        CS_FREE_HD(secvif);
        CS_MALLOC_HD(secvif, n_i_faces, cs_real_t, cs_alloc_mode);
      }

      /* Resize and reinitialize main internal faces properties array */
      cs_turbomachinery_reinit_i_face_fields();

      /* Update local pointers on "internal faces" fields */
      imasfl = cs_field_by_id(iflmas)->val;

      if (cs_glob_mesh->halo != nullptr) {

        cs_turbomachinery_resize_cell_fields();

        /* Update field mappings
           ("owner" fields handled by cs_turbomachinery_update);
           Remark: most of what is done in this call is redundant with the
           original initialization, and this call could probably be removed. */

        /* BC's do not need to be remapped as boundary faces are
           not expected to change */

        dt = cs_field_by_name("dt")->val;

        /* Resize auxiliary arrays (pointe module) */
        cs_fortran_resize_aux_arrays();

        /* Resize other arrays related to the velocity-pressure resolution */
        CS_REALLOC_HD(da_uu, n_cells_ext, cs_real_6_t, cs_alloc_mode);
        cs_mesh_sync_var_sym_tens(da_uu);

        CS_REALLOC_HD(trav, n_cells_ext, cs_real_3_t, cs_alloc_mode);
        cs_mesh_sync_var_vect((cs_real_t *)trav);

        CS_REALLOC_HD(dfrcxt, n_cells_ext, cs_real_3_t, cs_alloc_mode);
        cs_mesh_sync_var_vect((cs_real_t *)dfrcxt);

        /* Resize other arrays, depending on user options */

        if (   cs_glob_lagr_time_scheme->iilagr != CS_LAGR_OFF
            && cs_glob_lagr_dim->ntersl > 0) {
          _resize_non_interleaved_cell_arrays
            (m,
             cs_glob_lagr_dim->ntersl,
             &(cs_glob_lagr_source_terms->st_val));
        }

        if (vp_param->iphydr == 1)
          frcxt = (cs_real_3_t *)cs_field_by_name("volume_forces")->val;
        else if (vp_param->iphydr == 2) {
          CS_REALLOC_HD(grdphd, n_cells_ext, cs_real_3_t, cs_alloc_mode);
          cs_mesh_sync_var_vect((cs_real_t *)grdphd);
        }

        /* Update local pointers on "cells" fields */

        crom = CS_F_(rho)->val;
        crom_eos = CS_F_(rho)->val;

        if (   fluid_props->irovar == 1
            && (   vp_model->idilat > 1
                || vof_param->vof_model > 0
                || cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 3)) {

          cpro_rho_mass = cs_field_by_name("density_mass")->val;

          /* Time interpolated density */
          if (theta < 1.0 && vp_param->itpcol == 0) {
            croma = CS_F_(rho)->val_pre;
            CS_REALLOC_HD(cpro_rho_tc, n_cells_ext, cs_real_t, cs_alloc_mode);

            ctx.parallel_for(n_cells_ext, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
              cpro_rho_tc[c_id] =   theta * cpro_rho_mass[c_id]
                                  + (1.0 - theta) * croma[c_id];
            });

            crom = cpro_rho_tc;
            cromk1 = cpro_rho_tc;
          }
          else {
            crom = cpro_rho_mass;
            /* rho at time n+1,k-1 */
            CS_REALLOC_HD(cpro_rho_k1, n_cells_ext, cs_real_t, cs_alloc_mode);

            cs_array_copy<cs_real_t>(n_cells_ext, cpro_rho_mass, cpro_rho_k1);

            ctx.wait();

            cromk1 = cpro_rho_k1;
          }

        }
        else {
          crom = crom_eos;
          cromk1 = crom_eos; /* rho at time n+1,k-1 */
        }

        viscl = CS_F_(mu)->val;
        visct = CS_F_(mu_t)->val;

        vel = (cs_real_3_t *)CS_F_(vel)->val;
        vela = (cs_real_3_t *)CS_F_(vel)->val_pre;

        cvar_pr = CS_F_(p)->val;

        if (f_dttens != nullptr)
          dttens = (cs_real_6_t *)f_dttens->val;

        if (vp_param->nterup > 1) {
          CS_REALLOC_HD(velk, n_cells_ext, cs_real_3_t, cs_alloc_mode);
          cs_mesh_sync_var_vect((cs_real_t *)velk);
          CS_REALLOC_HD(trava, n_cells_ext, cs_real_3_t, cs_alloc_mode);
          cs_mesh_sync_var_vect((cs_real_t *)trava);
        }
        else {
          velk = vela;
        }

      } /* halo != nullptr */

    } /* cs_turbomachinery_get_n_couplings() < 1 */

    /* Update the Dirichlet wall boundary conditions for velocity (based on the
     * solid body rotation on the new mesh).
     * Note that the velocity BC update is made only if the user has
     * not specified any specific Dirichlet condition for velocity. */

    cs_real_t *coftur = nullptr,  *hfltur = nullptr;
    cs_turbomachinery_get_wall_bc_coeffs(&coftur, &hfltur);
    const int *irotce = cs_turbomachinery_get_cell_rotor_num();

    const cs_real_3_t *restrict b_face_u_normal
      = (const cs_real_3_t *restrict )mq->b_face_u_normal;
    const cs_real_3_t *restrict b_face_cog
      = (const cs_real_3_t *restrict)mq->b_face_cog;

    for (cs_lnum_t face_id = 0; face_id < n_b_faces; face_id++) {

      const cs_lnum_t c_id = b_face_cells[face_id];

      if (coftur[face_id] >= cs_math_infinite_r*0.5)
        continue;

      /* Physical Propreties */
      const cs_real_t visclc = viscl[c_id];
      const cs_real_t visctc = visct[c_id];

      /* Geometrical quantities */
      const cs_real_t distbf = mq->b_dist[face_id];

      /* Unit normal */
      const cs_real_t *ufn = b_face_u_normal[face_id];

      cs_real_t hint;
      if (cs_glob_turb_model->itytur == 3)
        hint = visclc / distbf;
      else
        hint = (visclc+visctc) / distbf;

      cs_real_t vr[3];
      cs_rotation_velocity(cs_glob_rotation + irotce[c_id],
                           b_face_cog[face_id],
                           vr);

      /* Gradient boundary conditions (Dirichlet) */
      const cs_real_t vrn = cs_math_3_dot_product(vr, ufn);

      for (cs_lnum_t ii = 0; ii < 3; ii++)
        coefau[face_id][ii] =   (1. - coftur[face_id]) * (vr[ii] - vrn*ufn[ii])
                              + vrn*ufn[ii];

      /* Flux boundary conditions (Dirichlet) */
      for (cs_lnum_t ii = 0; ii < 3; ii++)
        cofafu[face_id][ii] = -hfltur[face_id] * (vr[ii] - vrn*ufn[ii])
                              -hint*vrn*ufn[ii];
    }

    const cs_real_t t2 = cs_timer_wtime();

    rs_ell[1] = t2 - t1;

  } /* (iterns == 1) &&
       (cs_turbomachinery_get_model() == CS_TURBOMACHINERY_TRANSIENT) */

  /* Pressure correction step
     ------------------------ */

  if (eqp_u->verbosity > 0)
    bft_printf("** SOLVING CONTINUITY PRESSURE\n");

  cs_real_t *coefa_dp = cs_field_by_name("pressure_increment")->bc_coeffs->a;

  /* Pointers to BC coefficients */
  coefau = (cs_real_3_t *)CS_F_(vel)->bc_coeffs->a;

  /* Pressure correction step */
  if (   cs_glob_physical_model_flag[CS_COMPRESSIBLE] < 0
      || cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 3) {
    cs_pressure_correction(iterns,
                           w_condensation->nfbpcd,
                           w_condensation->ncmast,
                           w_condensation->ifbpcd,
                           w_condensation->ltmast,
                           isostd,
                           vel,
                           da_uu,
                           CS_F_(vel)->bc_coeffs,
                           cs_field_by_name("pressure_increment")->bc_coeffs,
                           w_condensation->spcond,
                           w_condensation->svcond,
                           frcxt,
                           dfrcxt,
                           viscf,
                           viscb);
  }

  /* Bad cells regularisation */
  cs_bad_cells_regularisation_scalar(cvar_pr);

  /* Update local pointers on "cells" fields */
  crom = CS_F_(rho)->val;
  crom_eos = CS_F_(rho)->val;

  /* Update density which may be computed in the pressure step */

  if (   fluid_props->irovar == 1
      && (   vp_model->idilat > 1
          || vof_param->vof_model > 0
          || cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 3)) {

    cpro_rho_mass = cs_field_by_name("density_mass")->val;

    /* Time interpolated density */
    if (theta < 1.0 && vp_param->itpcol == 0) {

      croma = CS_F_(rho)->val_pre;

      if (cpro_rho_tc != nullptr) {
        CS_FREE_HD(cpro_rho_tc);
        CS_MALLOC_HD(cpro_rho_tc, n_cells_ext, cs_real_t, cs_alloc_mode);
      }
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        cpro_rho_tc[c_id] =          theta * cpro_rho_mass[c_id]
                            + (1.0 - theta) * croma[c_id];
      });

      ctx.wait(); // needed for the following synchronization

      cs_mesh_sync_var_scal(cpro_rho_tc);

      crom = cpro_rho_tc;
      cromk1 = cpro_rho_tc; /* rho at time n+1/2,k-1 */
    }

    else
      crom = cpro_rho_mass;

  }

  /* Mesh velocity solving (ALE) */

  if (cs_glob_ale > CS_ALE_NONE) {
    if (itrale > cs_glob_ale_n_ini_f)
      cs_ale_solve_mesh_velocity(iterns);
  }

  /* Update of the fluid velocity field
     ---------------------------------- */

  if (   cs_glob_physical_model_flag[CS_COMPRESSIBLE] < 0
      || cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 3)
    _update_fluid_vel(m,
                      mq,
                      eqp_p,
                      vof_param,
                      dt,
                      crom,
                      cromk1,
                      imasfl,
                      bmasfl,
                      coefa_dp,
                      vel,
                      dfrcxt,
                      frcxt,
                      dttens,
                      isostd);

  /* Bad cells regularisation */
  cs_bad_cells_regularisation_vector(vel, 1);

  /* Mass flux initialization for VOF algorithm */
  if (vof_param->vof_model > 0) {
    cs_arrays_set_value<cs_real_t, 1>(n_i_faces, 0., imasfl);
    cs_arrays_set_value<cs_real_t, 1>(n_b_faces, 0., bmasfl);
  }

  /* In the ALE framework, we add the mesh velocity */
  if (cs_glob_ale > CS_ALE_NONE)
    _mesh_velocity_mass_flux(m, mq,
                             dt,
                             crom, brom,
                             imasfl, bmasfl);

  /* FIXME for me we should do that before cs_velocity_prediction */
  /* Add solid's velocity in convective flux if the mesh is mobile (rigid solid).
   * For turbomachinery, the mesh velocity to add is known exactly */

  if (cs_turbomachinery_get_model() > CS_TURBOMACHINERY_NONE) {
    const cs_real_t t3 = cs_timer_wtime();
    _turbomachinery_mass_flux(m, mq,
                              crom, brom,
                              imasfl, bmasfl);
    rs_ell[1] += cs_timer_wtime() - t3;
  }

  /* VoF: void fraction solving and update the mixture density/viscosity and
   *      mass flux (cs_pressure_correction solved the convective flux of
   *      void fraction, divU)
   * ------------------------------------------------------------------------ */

  if (vof_param->vof_model > 0) {

    /* Void fraction solving */
    cs_vof_solve_void_fraction(iterns);

    /* Halo synchronization */
    cs_real_t *cvar_voidf = cs_field_by_name("void_fraction")->val;
    cs_mesh_sync_var_scal(cvar_voidf);

    /* Update mixture density/viscosity and mass flux */
    cs_vof_update_phys_prop(m);

    /* Logging */
    if (iterns == vp_param->nterup && cs_log_default_is_active())
      cs_vof_log_mass_budget(m, mq);
  }

  /* Update density (which is coherent with the mass) */

  if (   fluid_props->irovar == 1
      && (   vp_model->idilat > 1
          || vof_param->vof_model > 0
          || cs_glob_physical_model_flag[CS_COMPRESSIBLE] == 3)) {

    ctx.parallel_for(n_cells_ext, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
      cpro_rho_mass[c_id] = crom_eos[c_id];
      crom[c_id] = crom_eos[c_id];
    });
    cs_array_copy<cs_real_t>(n_b_faces, brom_eos, bpro_rho_mass);
  }

  /* Compute error estimators for correction step and the global algorithm
     --------------------------------------------------------------------- */

  cs_field_t *iescor = cs_field_by_name_try("est_error_cor_2");
  cs_field_t *iestot = cs_field_by_name_try("est_error_tot_2");

  if (iescor != nullptr || iestot != nullptr) {

    const cs_real_t *cell_f_vol = mq->cell_f_vol;

    cs_real_t *esflum = nullptr, *esflub = nullptr;
    CS_MALLOC_HD(esflum, n_i_faces, cs_real_t, cs_alloc_mode);
    CS_MALLOC_HD(esflub, n_b_faces, cs_real_t, cs_alloc_mode);

    cs_mesh_sync_var_vect((cs_real_t *)vel);

    if (iestot != nullptr)
      cs_mesh_sync_var_scal(cvar_pr);

    int iflmb0 = 1;
    if (cs_glob_ale > CS_ALE_NONE)
      iflmb0 = 0;

    /* Mass flux based on updated velocity */

    cs_mass_flux(m,
                 mq,
                 CS_F_(vel)->id,
                 1,  /* itypfl */
                 iflmb0,
                 1,  /* init */
                 1,  /* inc */
                 eqp_u->imrgra,
                 eqp_u->nswrgr,
                 static_cast<cs_gradient_limit_t>(eqp_u->imligr),
                 eqp_u->verbosity,
                 eqp_u->epsrgr,
                 eqp_u->climgr,
                 crom, brom,
                 vel,
                 bc_coeffs_vel,
                 esflum , esflub);

    /* Correction estimator: div(rom * U(n + 1)) - gamma */

    if (iescor != nullptr) {
      cs_real_t *c_estim = iescor->val;
      cs_divergence(m, 1, esflum, esflub, c_estim);

      cs_lnum_t ncetsm = 0;
      const cs_lnum_t *icetsm = nullptr;
      cs_real_t *smacel = nullptr;
      cs_volume_mass_injection_get_arrays(CS_F_(p), &ncetsm, &icetsm, nullptr,
                                          &smacel, nullptr);

      if (ncetsm > 0) {

        ctx.parallel_for(ncetsm, [=] CS_F_HOST_DEVICE (cs_lnum_t c_idx) {
          cs_lnum_t c_id = icetsm[c_idx];
          c_estim[c_id] -= cell_f_vol[c_id] * smacel[c_idx];
        });
      }

      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        c_estim[c_id] = cs_math_fabs(c_estim[c_id]) / cell_f_vol[c_id];
      });
    }

    /* Total estimator */

    if (iestot != nullptr) {
      ctx.parallel_for(n_cells, [=] CS_F_HOST_DEVICE (cs_lnum_t c_id) {
        const cs_real_t rovolsdt = crom[c_id] * cell_f_vol[c_id] / dt[c_id];
        for (cs_lnum_t isou = 0; isou < 3; isou++)
          trav[c_id][isou] = rovolsdt * (vela[c_id][isou] - vel[c_id][isou]);
      });

      if (vp_param->staggered == 0) {
        _velocity_prediction(m,
                             mq,
                             2,
                             iterns,
                             dt,
                             vel,
                             vel,
                             velk,
                             da_uu,
                             bc_coeffs_vel,
                             ckupdc,
                             frcxt,
                             grdphd,
                             gxyz,
                             xyzp0,
                             trava,
                             dfrcxt,
                             dttens,
                             trav,
                             viscf,
                             viscb,
                             viscfi,
                             viscbi,
                             secvif,
                             secvib);
      }
    }

    CS_FREE_HD(esflum);
    CS_FREE_HD(esflub);

  }

  /* Velocity/pressure inner iterations
     ---------------------------------- */

  if (vp_param->nterup > 1) {

    /* Convergence test on U/P inner iterations, icvrge is 1 if converged */
    *icvrge = 1;

    const cs_real_t *cell_f_vol = mq->cell_f_vol;

    cs_real_t xnrtmp = 0;
#   pragma omp parallel for reduction(+:xnrtmp) if(n_cells > CS_THR_MIN)
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      cs_real_t xduvw[3] = {vel[c_id][0] - velk[c_id][0],
                            vel[c_id][1] - velk[c_id][1],
                            vel[c_id][2] - velk[c_id][2]};
      xnrtmp += cs_math_3_dot_product(xduvw, xduvw) * cell_f_vol[c_id];
    }
    cs_parall_sum(1, CS_REAL_TYPE, &xnrtmp);
    vp_param->xnrmu = xnrtmp;

    cs_real_t xnr_mu[] = {vp_param->xnrmu};
    for (int cpl_id = 0; cpl_id < nbrcpl; cpl_id++) {
      cs_real_t xnrdis[1];
      cs_sat_coupling_array_exchange(cpl_id,
                                     1, /* nbrdis */
                                     1, /* nbrloc */
                                     xnr_mu,
                                     xnrdis);
      xnr_mu[0] += xnrdis[0];
    }
    vp_param->xnrmu = sqrt(xnr_mu[0]);

    /* Fixed-point convergence indicator */
    if (vp_param->xnrmu >= vp_param->epsup * vp_param->xnrmu0)
      *icvrge = 0;

  }

  /* Shift pressure field to set its spatial mean value to zero
   * if there is no boundary faces with a Dirichlet condition on the pressure.
   * Number of faces with Dirichlet condition for the pressure is:
   * - ndircl if idiricl = 1
   * - ndircl-1 if idircl = 0 */

  int ndircp = 0;
  if (eqp_p->ndircl == 1)
    ndircp = eqp_p->ndircl;
  else
    ndircp = eqp_p->ndircl - 1;
  if (ndircp <= 0)
    cs_field_set_volume_average(CS_F_(p), fluid_props->pred0);

  /* Compute the total pressure (defined as a post-processed property).
   * For the compressible module, the solved pressure is already the
   * total pressure.
   * Remark: for Eddy Viscosity Models,
   *         TKE might be included in the solved pressure. */

  if (cs_glob_physical_model_flag[CS_COMPRESSIBLE] < 0)
    cs_solve_navier_stokes_update_total_pressure(m, mq, fluid_props,
                                                 gxyz, xyzp0);

  if (eqp_u->verbosity > 0)
    _log_norm(m, mq,
              iterns,
              *icvrge,
              crom, brom,
              imasfl, bmasfl,
              cvar_pr,
              vel);

  if (cs_turbomachinery_get_model() == CS_TURBOMACHINERY_TRANSIENT) {
    if (iterns == vp_param->nterup && cs_log_default_is_active())
      bft_printf("** INFORMATION ON UNSTEADY ROTOR/STATOR TREATMENT\n"
                 "   ----------------------------------------------\n"
                 " Time dedicated to mesh update (s): %10.4lf         \n"
                 " Global time                   (s): %10.4lf\n\n", rs_ell[0],
                 rs_ell[0] + rs_ell[1]);
  }

  CS_FREE_HD(trav);
  CS_FREE_HD(da_uu);
  CS_FREE_HD(dfrcxt);

  if (iterns == vp_param->nterup)
    CS_FREE_HD(trava);

  CS_FREE_HD(secvib);
  CS_FREE_HD(secvif);

  CS_FREE_HD(grdphd);

  CS_FREE_HD(bpro_rho_tc);
  CS_FREE_HD(cpro_rho_tc);

  CS_FREE_HD(wvisbi);
  CS_FREE_HD(wvisfi);

  CS_FREE_HD(uvwk);

  CS_FREE_HD(viscb);
  CS_FREE_HD(viscf);

  CS_FREE_HD(cpro_rho_k1);

#if defined(HAVE_ACCEL)
  CS_FREE_HD(_gxyz);
  CS_FREE_HD(_xyzp0);
#endif
}

/*----------------------------------------------------------------------------*/

END_C_DECLS

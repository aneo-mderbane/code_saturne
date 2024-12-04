#ifndef __CS_CDOFB_SCALEQ_H__
#define __CS_CDOFB_SCALEQ_H__

/*============================================================================
 * Build an algebraic CDO face-based system for unsteady convection/diffusion
 * reaction of scalar-valued equations with source terms
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

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "cs_base.h"
#include "cs_cdo_connect.h"
#include "cs_cdo_local.h"
#include "cs_cdo_toolbox.h"
#include "cs_cdo_quantities.h"
#include "cs_cdofb_priv.h"
#include "cs_equation_builder.h"
#include "cs_equation_param.h"
#include "cs_field.h"
#include "cs_matrix.h"
#include "cs_mesh.h"
#include "cs_restart.h"
#include "cs_source_term.h"
#include "cs_time_step.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*============================================================================
 * Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definitions
 *============================================================================*/

/* Algebraic system for CDO face-based discretization */
typedef struct _cs_cdofb_t cs_cdofb_scaleq_t;

/*============================================================================
 * Public function prototypes
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief    Check if the generic structures for building a CDO-Fb scheme are
 *           allocated
 *
 * \return  true or false
 */
/*----------------------------------------------------------------------------*/

bool
cs_cdofb_scaleq_is_initialized(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Allocate work buffer and general structures related to CDO
 *         scalar-valued face-based schemes.
 *         Set shared pointers from the main domain members
 *
 * \param[in]  quant       additional mesh quantities struct.
 * \param[in]  connect     pointer to a cs_cdo_connect_t struct.
 * \param[in]  time_step   pointer to a time step structure
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_init_sharing(const cs_cdo_quantities_t     *quant,
                             const cs_cdo_connect_t        *connect,
                             const cs_time_step_t          *time_step);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Retrieve work buffers used for building a CDO system cellwise
 *
 * \param[out]  csys   pointer to a pointer on a cs_cell_sys_t structure
 * \param[out]  cb     pointer to a pointer on a cs_cell_builder_t structure
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_get(cs_cell_sys_t       **csys,
                    cs_cell_builder_t   **cb);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Free work buffer and general structure related to CDO face-based
 *         schemes
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_finalize_sharing(void);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Initialize a cs_cdofb_scaleq_t structure storing data useful for
 *        building and managing such a scheme
 *
 * \param[in, out] eqp       set of parameters related an equation
 * \param[in]      var_id    id of the variable field
 * \param[in]      bflux_id  id of the boundary flux field
 * \param[in, out] eqb       pointer to a \ref cs_equation_builder_t structure
 *
 * \return a pointer to a new allocated cs_cdofb_scaleq_t structure
 */
/*----------------------------------------------------------------------------*/

void *
cs_cdofb_scaleq_init_context(cs_equation_param_t    *eqp,
                             int                     var_id,
                             int                     bflux_id,
                             cs_equation_builder_t  *eqb);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Destroy a cs_cdofb_scaleq_t structure
 *
 * \param[in, out]  data   pointer to a cs_cdofb_scaleq_t structure
 *
 * \return a null pointer
 */
/*----------------------------------------------------------------------------*/

void *
cs_cdofb_scaleq_free_context(void   *data);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Set the initial values of the variable field taking into account
 *         the boundary conditions.
 *         Case of scalar-valued CDO-Fb schemes.
 *
 * \param[in]      t_eval     time at which one evaluates BCs
 * \param[in]      field_id   id related to the variable field of this equation
 * \param[in]      mesh       pointer to a cs_mesh_t structure
 * \param[in]      eqp        pointer to a cs_equation_param_t structure
 * \param[in, out] eqb        pointer to a cs_equation_builder_t structure
 * \param[in, out] context    pointer to the scheme context (cast on-the-fly)
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_init_values(cs_real_t                     t_eval,
                            const int                     field_id,
                            const cs_mesh_t              *mesh,
                            const cs_equation_param_t    *eqp,
                            cs_equation_builder_t        *eqb,
                            void                         *context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Build and solve the linear system arising from a scalar steady-state
 *         convection/diffusion/reaction equation with a CDO-Fb scheme
 *         Use for interpolation purpose from cell values to face values.
 *         One works cellwise and then process to the assembly
 *
 * \param[in]      mesh         pointer to a cs_mesh_t structure
 * \param[in]      cell_values  array of cell values
 * \param[in]      field_id     id of the variable field
 * \param[in]      eqp          pointer to a cs_equation_param_t structure
 * \param[in, out] eqb          pointer to a cs_equation_builder_t structure
 * \param[in, out] context      pointer to cs_cdofb_scaleq_t structure
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_interpolate(const cs_mesh_t            *mesh,
                            const cs_real_t            *cell_values,
                            const int                   field_id,
                            const cs_equation_param_t  *eqp,
                            cs_equation_builder_t      *eqb,
                            void                       *context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Build and solve the linear system arising from a scalar steady-state
 *         convection/diffusion/reaction equation with a CDO-Fb scheme
 *         One works cellwise and then process to the assembly
 *
 * \param[in]      cur2prev   true="current to previous" operation is performed
 * \param[in]      mesh       pointer to a cs_mesh_t structure
 * \param[in]      field_id   id of the variable field related to this equation
 * \param[in]      eqp        pointer to a cs_equation_param_t structure
 * \param[in, out] eqb        pointer to a cs_equation_builder_t structure
 * \param[in, out] context    pointer to cs_cdofb_scaleq_t structure
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_solve_steady_state(bool                        cur2prev,
                                   const cs_mesh_t            *mesh,
                                   const int                   field_id,
                                   const cs_equation_param_t  *eqp,
                                   cs_equation_builder_t      *eqb,
                                   void                       *context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Build and solve the linear system arising from a scalar
 *         convection/diffusion/reaction equation with a CDO-Fb scheme and an
 *         implicit Euler scheme.
 *         One works cellwise and then process to the assembly
 *
 * \param[in]      cur2prev   true="current to previous" operation is performed
 * \param[in]      mesh       pointer to a cs_mesh_t structure
 * \param[in]      field_id   id of the variable field related to this equation
 * \param[in]      eqp        pointer to a cs_equation_param_t structure
 * \param[in, out] eqb        pointer to a cs_equation_builder_t structure
 * \param[in, out] context    pointer to cs_cdofb_scaleq_t structure
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_solve_implicit(bool                        cur2prev,
                               const cs_mesh_t            *mesh,
                               const int                   field_id,
                               const cs_equation_param_t  *eqp,
                               cs_equation_builder_t      *eqb,
                               void                       *context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Build and solve the linear system arising from a scalar
 *         convection/diffusion/reaction equation with a CDO-Fb scheme and an
 *         implicit/explicit theta scheme.
 *         One works cellwise and then process to the assembly
 *
 * \param[in]      cur2prev   true="current to previous" operation is performed
 * \param[in]      mesh       pointer to a cs_mesh_t structure
 * \param[in]      field_id   id of the variable field related to this equation
 * \param[in]      eqp        pointer to a cs_equation_param_t structure
 * \param[in, out] eqb        pointer to a cs_equation_builder_t structure
 * \param[in, out] context    pointer to cs_cdofb_scaleq_t structure
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_solve_theta(bool                        cur2prev,
                            const cs_mesh_t            *mesh,
                            const int                   field_id,
                            const cs_equation_param_t  *eqp,
                            cs_equation_builder_t      *eqb,
                            void                       *context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Compute the balance for an equation over the full computational
 *         domain between time t_cur and t_cur + dt_cur
 *         Case of scalar-valued CDO face-based scheme
 *
 * \param[in]      eqp       pointer to a \ref cs_equation_param_t structure
 * \param[in, out] eqb       pointer to a \ref cs_equation_builder_t structure
 * \param[in, out] context   pointer to a scheme builder structure
 *
 * \return a pointer to a \ref cs_cdo_balance_t structure
 */
/*----------------------------------------------------------------------------*/

cs_cdo_balance_t *
cs_cdofb_scaleq_balance(const cs_equation_param_t     *eqp,
                        cs_equation_builder_t         *eqb,
                        void                          *context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Cellwise computation of the diffusive flux accross primal faces.
 *        Interior faces first and then boundary faces.
 *        Values at faces are recovered thanks to the equation builder
 *        Case of scalar-valued CDO-Fb schemes
 *
 * \param[in]      f_values    values for the potential at faces
 * \param[in]      c_values    values for the potential at cells
 * \param[in]      eqp         pointer to a cs_equation_param_t structure
 * \param[in]      diff_pty    pointer to the diffusion property to use
 * \param[in]      t_eval      time at which one performs the evaluation
 * \param[in, out] eqb         pointer to a cs_equation_builder_t structure
 * \param[in, out] context     pointer to cs_cdovb_scaleq_t structure
 * \param[in, out] diff_flux   value of the diffusive flux at primal faces
  */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_diff_flux_faces(const cs_real_t             *f_values,
                                const cs_real_t             *c_values,
                                const cs_equation_param_t   *eqp,
                                const cs_property_t         *diff_pty,
                                cs_real_t                    t_eval,
                                cs_equation_builder_t       *eqb,
                                void                        *context,
                                cs_real_t                   *diff_flux);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute an approximation of the the diffusive flux across each
 *        boundary face.
 *        Case of scalar-valued CDO-Fb schemes
 *
 * \param[in]      pot_f     array of values at faces
 * \param[in]      pot_c     array of values at cells
 * \param[in]      eqp       pointer to a cs_equation_param_t structure
 * \param[in]      diff_pty  pointer to the diffusion property to use
 * \param[in]      t_eval    time at which one performs the evaluation
 * \param[in, out] eqb       pointer to a cs_equation_builder_t structure
 * \param[in, out] context   pointer to a scheme builder structure
 * \param[in, out] bflux     pointer to the values of the diffusive flux
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_boundary_diff_flux(const cs_real_t           *pot_f,
                                   const cs_real_t           *pot_c,
                                   const cs_equation_param_t *eqp,
                                   const cs_property_t       *diff_pty,
                                   const cs_real_t            t_eval,
                                   cs_equation_builder_t     *eqb,
                                   void                      *context,
                                   cs_real_t                 *bflux);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Operate a current to previous operation for the field associated to
 *         this equation and potentially for related fields/arrays.
 *
 * \param[in]       eqp        pointer to a cs_equation_param_t structure
 * \param[in, out]  eqb        pointer to a cs_equation_builder_t structure
 * \param[in, out]  context    pointer to cs_cdofb_scaleq_t structure
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_current_to_previous(const cs_equation_param_t  *eqp,
                                    cs_equation_builder_t      *eqb,
                                    void                       *context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Predefined extra-operations related to this equation
 *
 * \param[in]       eqp        pointer to a cs_equation_param_t structure
 * \param[in, out]  eqb        pointer to a cs_equation_builder_t structure
 * \param[in, out]  context    pointer to cs_cdofb_scaleq_t structure
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_extra_post(const cs_equation_param_t  *eqp,
                           cs_equation_builder_t      *eqb,
                           void                       *context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Get the computed values at mesh cells from the inverse operation
 *         w.r.t. the static condensation (DoF used in the linear system are
 *         located at primal faces)
 *         The lifecycle of this array is managed by the code. So one does not
 *         have to free the return pointer.
 *
 * \param[in, out]  context    pointer to a data structure cast on-the-fly
 * \param[in]       previous   retrieve the previous state (true/false)
 *
 * \return  a pointer to an array of cs_real_t (size n_cells)
 */
/*----------------------------------------------------------------------------*/

cs_real_t *
cs_cdofb_scaleq_get_cell_values(void      *context,
                                bool       previous);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Retrieve an array of values at mesh faces for the current context.
 *         The lifecycle of this array is managed by the code. So one does not
 *         have to free the return pointer.
 *
 * \param[in, out]  context    pointer to a data structure cast on-the-fly
 * \param[in]       previous   retrieve the previous state (true/false)
 *
 * \return  a pointer to an array of cs_real_t (size n_faces)
 */
/*----------------------------------------------------------------------------*/

cs_real_t *
cs_cdofb_scaleq_get_face_values(void    *context,
                                bool     previous);

/*----------------------------------------------------------------------------*/
/*!
 * \brief Retrieve the array storing the source term values at mesh cells.
 *        The lifecycle of this array is managed by the code. So one does not
 *        have to free the return pointer.
 *
 * \param[in, out] context    pointer to a data structure cast on-the-fly
 *
 * \return a pointer to an array of cs_real_t (size n_cells)
 */
/*----------------------------------------------------------------------------*/

cs_real_t *
cs_cdofb_scaleq_get_source_term_values(void    *context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Read additional arrays (not defined as fields) but useful for the
 *         checkpoint/restart process
 *
 * \param[in, out]  restart         pointer to \ref cs_restart_t structure
 * \param[in]       eqname          name of the related equation
 * \param[in]       scheme_context  pointer to a data structure cast on-the-fly
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_read_restart(cs_restart_t    *restart,
                             const char      *eqname,
                             void            *scheme_context);

/*----------------------------------------------------------------------------*/
/*!
 * \brief  Write additional arrays (not defined as fields) but useful for the
 *         checkpoint/restart process
 *
 * \param[in, out]  restart         pointer to \ref cs_restart_t structure
 * \param[in]       eqname          name of the related equation
 * \param[in]       scheme_context  pointer to a data structure cast on-the-fly
 */
/*----------------------------------------------------------------------------*/

void
cs_cdofb_scaleq_write_restart(cs_restart_t    *restart,
                              const char      *eqname,
                              void            *scheme_context);

/*----------------------------------------------------------------------------*/

END_C_DECLS

#endif /* __CS_CDOFB_SCALEQ_H__ */

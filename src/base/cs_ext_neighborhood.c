/*============================================================================
 *
 *     This file is part of the Code_Saturne Kernel, element of the
 *     Code_Saturne CFD tool.
 *
 *     Copyright (C) 1998-2008 EDF S.A., France
 *
 *     contact: saturne-support@edf.fr
 *
 *     The Code_Saturne Kernel is free software; you can redistribute it
 *     and/or modify it under the terms of the GNU General Public License
 *     as published by the Free Software Foundation; either version 2 of
 *     the License, or (at your option) any later version.
 *
 *     The Code_Saturne Kernel is distributed in the hope that it will be
 *     useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 *     of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with the Code_Saturne Kernel; if not, write to the
 *     Free Software Foundation, Inc.,
 *     51 Franklin St, Fifth Floor,
 *     Boston, MA  02110-1301  USA
 *
 *============================================================================*/

/*============================================================================
 * Fortran interfaces of functions needing a synchronization of the extended
 * neighborhood.
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <float.h>

#if defined(_CS_HAVE_MPI)
#include <mpi.h>
#endif

/*----------------------------------------------------------------------------
 * BFT library headers
 *----------------------------------------------------------------------------*/

#include <bft_error.h>
#include <bft_mem.h>
#include <bft_printf.h>

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "cs_halo.h"
#include "cs_mesh.h"
#include "cs_mesh_quantities.h"
#include "cs_perio.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "cs_ext_neighborhood.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Local Macro definitions
 *============================================================================*/

/*============================================================================
 * Type definition
 *============================================================================*/

/*============================================================================
 * Static global variables
 *============================================================================*/

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Extract a mesh's "cell -> internal faces" connectivity.
 *
 * parameters:
 *   mesh               --> pointer to a cs_mesh_t structure
 *   p_cell_i_faces_idx <-- pointer to the "cell -> faces" connectivity index
 *                          (1 to n numbering)
 *   p_cell_i_faces_lst <-- pointer to the "cell -> faces" connectivity list
 *----------------------------------------------------------------------------*/

static void
_get_cell_i_faces_connectivity(const cs_mesh_t          *mesh,
                               cs_int_t         * *const p_cell_i_faces_idx,
                               cs_int_t         * *const p_cell_i_faces_lst)
{

  cs_int_t  i, j, j1, j2;

  cs_int_t  *cell_faces_count = NULL;
  cs_int_t  *cell_faces_idx = NULL;
  cs_int_t  *cell_faces_lst = NULL;

  /* Allocate and initialize index */

  BFT_MALLOC(cell_faces_idx, mesh->n_cells + 1, cs_int_t);

  for (i = 0 ; i < mesh->n_cells + 1 ; i++)
    cell_faces_idx[i] = 0;

  /* Count number of faces per cell (we assign the temporary counter
   * to i and cell_faces_idx[i + 1] instead of cell_faces_idx[i]
   * to simplify the next stage) */

  /* Note: test if j < mesh->n_cells on internal faces to ignore
     parallel and/or periodic ghost cells */

  for (i = 0 ; i < mesh->n_i_faces ; i++) {
    j1 = mesh->i_face_cells[i*2    ] - 1;
    j2 = mesh->i_face_cells[i*2 + 1] - 1;
    if (j1 < mesh->n_cells)
      cell_faces_idx[j1 + 1] += 1;
    if (j2 < mesh->n_cells)
      cell_faces_idx[j2 + 1] += 1;
  }

  /* Build position index */

  cell_faces_idx[0] = 1;
  for (j = 0 ; j < mesh->n_cells ; j++)
    cell_faces_idx[j + 1] = cell_faces_idx[j] + cell_faces_idx[j + 1];

  /* Build array of values */

  BFT_MALLOC(cell_faces_lst, cell_faces_idx[mesh->n_cells] - 1, cs_int_t);
  BFT_MALLOC(cell_faces_count, mesh->n_cells, cs_int_t);

  for (i = 0 ; i < mesh->n_cells ; i++)
    cell_faces_count[i] = 0;

  for (i = 0 ; i < mesh->n_i_faces ; i++) {
    j1 = mesh->i_face_cells[i*2    ] - 1;
    j2 = mesh->i_face_cells[i*2 + 1] - 1;
    if (j1 < mesh->n_cells) {
      cell_faces_lst[cell_faces_idx[j1] + cell_faces_count[j1] - 1] = i + 1;
      cell_faces_count[j1] += 1;
    }
    if (j2 < mesh->n_cells) {
      cell_faces_lst[cell_faces_idx[j2] + cell_faces_count[j2] - 1] = -(i + 1);
      cell_faces_count[j2] += 1;
    }
  }

  BFT_FREE(cell_faces_count);

  /* Set return values */

  *p_cell_i_faces_idx = cell_faces_idx;
  *p_cell_i_faces_lst = cell_faces_lst;

}

/*----------------------------------------------------------------------------
 * Create a "vertex -> cells" connectivity.
 *
 * parameters:
 *   mesh            --> pointer to a cs_mesh_t structure
 *   p_vtx_cells_idx <-- pointer to the "vtx -> cells" connectivity index
 *   p_vtx_cells_lst <-- pointer to the "vtx -> cells" connectivity list
 *----------------------------------------------------------------------------*/

static void
_create_vtx_cells_connect(cs_mesh_t  *mesh,
                          cs_int_t   *p_vtx_cells_idx[],
                          cs_int_t   *p_vtx_cells_lst[])
{
  cs_int_t  i, j, idx;
  cs_int_t  vtx_id, face_id, cell_num;

  cs_bool_t  already_seen;

  cs_int_t  vtx_cells_connect_size = 0;

  cs_int_t  *vtx_faces_idx = NULL, *vtx_faces_lst = NULL;
  cs_int_t  *vtx_cells_idx = NULL, *vtx_cells_lst = NULL;

  const cs_int_t  n_vertices = mesh->n_vertices;
  const cs_int_t  n_faces = mesh->n_i_faces;
  const cs_int_t  *face_vtx_idx = mesh->i_face_vtx_idx;
  const cs_int_t  *face_vtx_lst = mesh->i_face_vtx_lst;
  const cs_int_t  *face_cells = mesh->i_face_cells;

  cs_int_t  vtx_cells_estimated_connect_size = 3 * n_vertices;

  BFT_MALLOC(vtx_cells_idx, n_vertices + 1, cs_int_t);
  BFT_MALLOC(vtx_faces_idx, n_vertices + 1, cs_int_t);

  for (vtx_id = 0; vtx_id < n_vertices + 1; vtx_id++) {
    vtx_cells_idx[vtx_id] = 0;
    vtx_faces_idx[vtx_id] = 0;
  }

  /* Define vtx -> faces connectivity index */

  for (face_id = 0; face_id < n_faces; face_id++) {

    for (i = face_vtx_idx[face_id] - 1; i < face_vtx_idx[face_id+1] - 1; i++) {
      vtx_id = face_vtx_lst[i] - 1;
      vtx_faces_idx[vtx_id + 1] += 1;
    }

  } /* End of loop on faces */

  vtx_faces_idx[0] = 1;
  for (vtx_id = 0; vtx_id < n_vertices; vtx_id++)
    vtx_faces_idx[vtx_id + 1] += vtx_faces_idx[vtx_id];

  /* Allocation and definiton of "vtx -> faces" connectivity list */

  BFT_MALLOC(vtx_faces_lst, vtx_faces_idx[n_vertices] - 1, cs_int_t);

  for (face_id = 0; face_id < n_faces; face_id++) {

    for (i = face_vtx_idx[face_id] - 1; i < face_vtx_idx[face_id+1] - 1; i++) {

      vtx_id = face_vtx_lst[i] - 1;
      vtx_faces_lst[vtx_faces_idx[vtx_id]-1] = face_id + 1;
      vtx_faces_idx[vtx_id] += 1;

    }

  } /* End of loop on faces */

  for (vtx_id = n_vertices; vtx_id > 0; vtx_id--)
    vtx_faces_idx[vtx_id] = vtx_faces_idx[vtx_id-1];
  vtx_faces_idx[0] = 1;

  /* Define "vertex -> cells" connectivity.
     Use "vertex -> faces" connectivity and "face -> cells" connectivity */

  BFT_MALLOC(vtx_cells_lst, vtx_cells_estimated_connect_size, cs_int_t);

  vtx_cells_idx[0] = 1;

  for (vtx_id = 0; vtx_id < n_vertices; vtx_id++) {

    for (i = vtx_faces_idx[vtx_id] - 1; i < vtx_faces_idx[vtx_id+1] - 1; i++) {

      face_id = vtx_faces_lst[i] - 1;

      for (j = 0; j < 2; j++) { /* For the cells sharing this face */

        cell_num = face_cells[2*face_id + j];

        already_seen = false;
        idx = vtx_cells_idx[vtx_id] - 1;

        while ((already_seen == false) && (idx < vtx_cells_connect_size)) {
          if (cell_num == vtx_cells_lst[idx])
            already_seen = true;
          idx++;
        }

        if (already_seen == false) {

          if (vtx_cells_connect_size + 1 > vtx_cells_estimated_connect_size) {
            vtx_cells_estimated_connect_size *= 2;
            BFT_REALLOC(vtx_cells_lst,
                        vtx_cells_estimated_connect_size, cs_int_t);
          }

          vtx_cells_lst[vtx_cells_connect_size] = cell_num;
          vtx_cells_connect_size += 1;

        }

      } /* End of loop on cells sharing this face */

    } /* End of loop on faces sharing this vertex */

    vtx_cells_idx[vtx_id+1] = vtx_cells_connect_size + 1;

  } /* End of loop on vertices */

  BFT_REALLOC(vtx_cells_lst, vtx_cells_connect_size, cs_int_t);

  /* Free memory */

  BFT_FREE(vtx_faces_idx);
  BFT_FREE(vtx_faces_lst);

  *p_vtx_cells_idx = vtx_cells_idx;
  *p_vtx_cells_lst = vtx_cells_lst;

}

/*----------------------------------------------------------------------------
 * Create a "vertex -> cells" connectivity.
 *
 * parameters:
 *   face_id        --> identification number for the face
 *   cell_id        --> identification number for the cell sharing this face
 *   mesh           --> pointer to a cs_mesh_t structure
 *   vtx_cells_idx  <-- pointer to the "vtx -> cells" connectivity index
 *   vtx_cells_lst  <-- pointer to the "vtx -> cells" connectivity list
 *----------------------------------------------------------------------------*/

static void
_tag_cells(cs_int_t    face_id,
           cs_int_t    cell_id,
           cs_mesh_t  *mesh,
           cs_int_t    vtx_cells_idx[],
           cs_int_t    vtx_cells_lst[])
{
  cs_int_t  i, j, k;
  cs_int_t  vtx_id, ext_cell_num, cell_num;

  cs_int_t  *cell_cells_lst = mesh->cell_cells_lst;
  cs_int_t  *cell_cells_idx = mesh->cell_cells_idx;

  const cs_int_t  n_cells = mesh->n_cells;
  const cs_int_t  *face_vtx_idx = mesh->i_face_vtx_idx;
  const cs_int_t  *face_vtx_lst = mesh->i_face_vtx_lst;

  if (cell_id < n_cells) {

    for (i = cell_cells_idx[cell_id] - 1;
         i < cell_cells_idx[cell_id+1] - 1; i++) {

      ext_cell_num = cell_cells_lst[i];

      /* Extended neighborhood not kept yet */

      if (ext_cell_num > 0) {

        /* Cells sharing a vertex with the face */

        for (j = face_vtx_idx[face_id] - 1;
             j < face_vtx_idx[face_id+1] - 1; j++) {

          vtx_id = face_vtx_lst[j] - 1;

          for (k = vtx_cells_idx[vtx_id] - 1;
               k < vtx_cells_idx[vtx_id+1] - 1; k++) {

            cell_num = vtx_cells_lst[k];

            /* Comparison and selection */

            if (cell_num == ext_cell_num && cell_cells_lst[i] > 0)
              cell_cells_lst[i] = - cell_cells_lst[i];

          } /* End of loop on cells sharing this vertex */

        } /* End of loop on vertices of the face */

      }

    } /* End of loop on cells in the extended neighborhood */

  } /* End if cell_id < n_cells */

}

/*---------------------------------------------------------------------------
 * Reverse "ghost cell -> vertex" connectivity into "vertex -> ghost cells"
 * connectivity for halo elements.
 * Build the connectivity index.
 *
 * parameters:
 *   halo            --> pointer to a cs_halo_t structure
 *   rank_id         --> rank number to work with
 *   checker         <-> temporary array to check vertices
 *   gcell_vtx_idx   --> "ghost cell -> vertices" connectivity index
 *   gcell_vtx_lst   --> "ghost cell -> vertices" connectivity list
 *   vtx_gcells_idx  <-> "vertex -> ghost cells" connectivity index
 *---------------------------------------------------------------------------*/

static void
_reverse_connectivity_idx(cs_halo_t  *halo,
                          cs_int_t    n_vertices,
                          cs_int_t    rank_id,
                          cs_int_t   *checker,
                          cs_int_t   *gcell_vtx_idx,
                          cs_int_t   *gcell_vtx_lst,
                          cs_int_t   *vtx_gcells_idx)
{
  cs_int_t  i, j, id, vtx_id, start_idx, end_idx;

  /* Initialize index */

  vtx_gcells_idx[0] = 0;
  for (i = 0; i < n_vertices; i++) {
    vtx_gcells_idx[i+1] = 0;
    checker[i] = -1;
  }

  if (rank_id == -1) {
    start_idx = 0;
    end_idx = halo->n_elts[CS_HALO_EXTENDED];
  }
  else { /* Call with rank_id > 1 for standard halo */
    start_idx = halo->index[2*rank_id];
    end_idx = halo->index[2*rank_id+1];
  }

  /* Define index */

  for (id = start_idx; id < end_idx; id++) {

    for (j = gcell_vtx_idx[id]; j < gcell_vtx_idx[id+1]; j++) {

      vtx_id = gcell_vtx_lst[j] - 1;

      if (checker[vtx_id] != id) {
        checker[vtx_id] = id;
        vtx_gcells_idx[vtx_id+1] += 1;
      }

    }

  } /* End of loop of ghost cells */

  for (i = 0; i < n_vertices; i++)
    vtx_gcells_idx[i+1] += vtx_gcells_idx[i];

}

/*---------------------------------------------------------------------------
 * Reverse "ghost cells -> vertex" connectivity into "vertex -> ghost cells"
 * connectivity for halo elements.
 * Build the connectivity list.
 *
 * parameters:
 *   halo            --> pointer to a cs_halo_t structure
 *   n_vertices      --> number of vertices
 *   rank_id         --> rank number to work with
 *   counter         <-> temporary array to count vertices
 *   checker         <-> temporary array to check vertices
 *   gcell_vtx_idx   --> "ghost cell -> vertices" connectivity index
 *   gcell_vtx_lst   --> "ghost cell -> vertices" connectivity list
 *   vtx_gcells_idx  --> "vertex -> ghost cells" connectivity index
 *   vtx_gcells_lst  <-> "vertex -> ghost cells" connectivity list
 *---------------------------------------------------------------------------*/

static void
_reverse_connectivity_lst(cs_halo_t  *halo,
                          cs_int_t    n_vertices,
                          cs_int_t    rank_id,
                          cs_int_t   *counter,
                          cs_int_t   *checker,
                          cs_int_t   *gcell_vtx_idx,
                          cs_int_t   *gcell_vtx_lst,
                          cs_int_t   *vtx_gcells_idx,
                          cs_int_t   *vtx_gcells_lst)
{
  cs_int_t  i, j, id, shift, vtx_id, start_idx, end_idx;

  /* Initialize buffers */

  for (i = 0; i < n_vertices; i++) {
    counter[i] = 0;
    checker[i] = -1;
  }

  if (rank_id == -1) {
    start_idx = 0;
    end_idx = halo->n_elts[CS_HALO_EXTENDED];
  }
  else {
    start_idx = halo->index[2*rank_id];
    end_idx = halo->index[2*rank_id+1];
  }

  /* Fill the connectivity list */

  for (id = start_idx; id < end_idx; id++) {

    for (j = gcell_vtx_idx[id]; j < gcell_vtx_idx[id+1]; j++) {

      vtx_id = gcell_vtx_lst[j] - 1;

      if (checker[vtx_id] != id) {

        checker[vtx_id] = id;
        shift = vtx_gcells_idx[vtx_id] + counter[vtx_id];
        vtx_gcells_lst[shift] = id;
        counter[vtx_id] += 1;

      }

    }

  } /* End of loop of ghost cells */

}

/*---------------------------------------------------------------------------
 * Create a "vertex -> ghost cells" connectivity.
 * Add mesh->n_cells to all the elements of vtx_gcells_lst to obtain
 * the local cell numbering.
 *
 * parameters:
 *   halo              --> pointer to a cs_halo_t structure
 *   n_vertices        --> number of vertices
 *   gcell_vtx_idx     --> "ghost cell -> vertices" connectivity index
 *   gcell_vtx_lst     --> "ghost cell -> vertices" connectivity list
 *   p_vtx_gcells_idx  <-- pointer to "vertex -> ghost cells" index
 *   p_vtx_gcells_lst  <-- pointer to "vertex -> ghost cells" list
 *---------------------------------------------------------------------------*/

static void
_create_vtx_gcells_connect(cs_halo_t   *halo,
                           cs_int_t     n_vertices,
                           cs_int_t    *gcells_vtx_idx,
                           cs_int_t    *gcells_vtx_lst,
                           cs_int_t    *p_vtx_gcells_idx[],
                           cs_int_t    *p_vtx_gcells_lst[])
{
  cs_int_t  *vtx_buffer = NULL, *vtx_counter = NULL, *vtx_checker = NULL;
  cs_int_t  *vtx_gcells_idx = NULL, *vtx_gcells_lst = NULL;

  BFT_MALLOC(vtx_buffer, 2*n_vertices, cs_int_t);
  vtx_counter = &(vtx_buffer[0]);
  vtx_checker = &(vtx_buffer[n_vertices]);

  BFT_MALLOC(vtx_gcells_idx, n_vertices + 1, cs_int_t);

  /* Create a vertex -> ghost cells connectivity */

  _reverse_connectivity_idx(halo,
                            n_vertices,
                            -1,
                            vtx_checker,
                            gcells_vtx_idx,
                            gcells_vtx_lst,
                            vtx_gcells_idx);

  BFT_MALLOC(vtx_gcells_lst, vtx_gcells_idx[n_vertices], cs_int_t);

  _reverse_connectivity_lst(halo,
                            n_vertices,
                            -1,
                            vtx_counter,
                            vtx_checker,
                            gcells_vtx_idx,
                            gcells_vtx_lst,
                            vtx_gcells_idx,
                            vtx_gcells_lst);

  *p_vtx_gcells_idx = vtx_gcells_idx;
  *p_vtx_gcells_lst = vtx_gcells_lst;

  /* Free memory */

  BFT_FREE(vtx_buffer);

}

/*---------------------------------------------------------------------------
 * Create a "vertex -> cells" connectivity.
 *
 * parameters:
 *   mesh              --> pointer to cs_mesh_t structure
 *   cell_i_faces_idx  --> "cell -> internal faces" connectivity index
 *   cell_i_faces_lst  --> "cell -> internal faces" connectivity list
 *   p_vtx_cells_idx   <-- pointer to "vertex -> cells" connectivity index
 *   p_vtx_cells_lst   <-- pointer to "vertex -> cells" connectivity list
 *---------------------------------------------------------------------------*/

static void
_create_vtx_cells_connect2(cs_mesh_t   *mesh,
                           cs_int_t    *cell_i_faces_idx,
                           cs_int_t    *cell_i_faces_lst,
                           cs_int_t    *p_vtx_cells_idx[],
                           cs_int_t    *p_vtx_cells_lst[])
{
  cs_int_t  i, cell_id, fac_id, i_vtx;
  cs_int_t  shift, vtx_id;

  cs_int_t  *vtx_buffer = NULL, *vtx_count = NULL, *vtx_tag = NULL;
  cs_int_t  *vtx_cells_idx = NULL, *vtx_cells_lst = NULL;

  const cs_int_t  n_cells = mesh->n_cells;
  const cs_int_t  n_vertices = mesh->n_vertices;
  const cs_int_t  *fac_vtx_idx = mesh->i_face_vtx_idx;
  const cs_int_t  *fac_vtx_lst = mesh->i_face_vtx_lst;

  /* Initialize buffers */

  BFT_MALLOC(vtx_buffer, 2*n_vertices, cs_int_t);
  BFT_MALLOC(vtx_cells_idx, n_vertices + 1, cs_int_t);

  vtx_count = &(vtx_buffer[0]);
  vtx_tag = &(vtx_buffer[n_vertices]);

  vtx_cells_idx[0] = 0;
  for (i = 0; i < n_vertices; i++) {

    vtx_cells_idx[i + 1] = 0;
    vtx_tag[i] = -1;
    vtx_count[i] = 0;

  }

  /* Define index */

  for (cell_id = 0; cell_id < n_cells; cell_id++) {

    for (i = cell_i_faces_idx[cell_id]; i < cell_i_faces_idx[cell_id+1]; i++) {

      fac_id = CS_ABS(cell_i_faces_lst[i-1]) - 1;

      for (i_vtx = fac_vtx_idx[fac_id];
           i_vtx < fac_vtx_idx[fac_id+1]; i_vtx++) {

        vtx_id = fac_vtx_lst[i_vtx-1] - 1;

        if (vtx_tag[vtx_id] != cell_id) {

          vtx_cells_idx[vtx_id+1] += 1;
          vtx_tag[vtx_id] = cell_id;

        } /* Add this cell to the connectivity of the vertex */

      } /* End of loop on vertices */

    } /* End of loop on cell's faces */

  } /* End of loop on cells */

  for (i = 0; i < n_vertices; i++) {

    vtx_cells_idx[i+1] += vtx_cells_idx[i];
    vtx_tag[i] = -1;

  }

  BFT_MALLOC(vtx_cells_lst, vtx_cells_idx[n_vertices], cs_int_t);

  /* Fill list */

  for (cell_id = 0; cell_id < n_cells; cell_id++) {

    for (i = cell_i_faces_idx[cell_id]; i < cell_i_faces_idx[cell_id+1]; i++) {

      fac_id = CS_ABS(cell_i_faces_lst[i-1]) - 1;

      for (i_vtx = fac_vtx_idx[fac_id];
           i_vtx < fac_vtx_idx[fac_id+1]; i_vtx++) {

        vtx_id = fac_vtx_lst[i_vtx-1] - 1;

        if (vtx_tag[vtx_id] != cell_id) {

          shift = vtx_cells_idx[vtx_id] + vtx_count[vtx_id];
          vtx_tag[vtx_id] = cell_id;
          vtx_cells_lst[shift] = cell_id;
          vtx_count[vtx_id] += 1;

        } /* Add this cell to the connectivity of the vertex */

      } /* End of loop on vertices */

    } /* End of loop on cell's faces */

  } /* End of loop on cells */

  BFT_FREE(vtx_buffer);

  *p_vtx_cells_idx = vtx_cells_idx;
  *p_vtx_cells_lst = vtx_cells_lst;

}

/*---------------------------------------------------------------------------
 * Create a "cell -> cells" connectivity.
 *
 * parameters:
 *   mesh              --> pointer to cs_mesh_t structure
 *   cell_i_faces_idx  --> "cell -> faces" connectivity index
 *   cell_i_faces_lst  --> "cell -> faces" connectivity list
 *   vtx_gcells_idx    <-- "vertex -> ghost cells" connectivity index
 *   vtx_gcells_lst    <-- "vertex -> ghost cells" connectivity list
 *   vtx_cells_idx     <-- "vertex -> cells" connectivity index
 *   vtx_cells_lst     <-- "vertex -> cells" connectivity list
 *   p_cell_cells_idx  <-- pointer to "cell -> cells" connectivity index
 *   p_cell_cells_lst  <-- pointer to "cell -> cells" connectivity list
 *---------------------------------------------------------------------------*/

static void
_create_cell_cells_connect(cs_mesh_t   *mesh,
                           cs_int_t    *cell_i_faces_idx,
                           cs_int_t    *cell_i_faces_lst,
                           cs_int_t    *vtx_gcells_idx,
                           cs_int_t    *vtx_gcells_lst,
                           cs_int_t    *vtx_cells_idx,
                           cs_int_t    *vtx_cells_lst,
                           cs_int_t    *p_cell_cells_idx[],
                           cs_int_t    *p_cell_cells_lst[])
{
  cs_int_t  i, j, i_cel, fac_id, i_vtx;
  cs_int_t  cell_id, vtx_id, shift;

  cs_int_t  *cell_buffer = NULL, *cell_tag = NULL, *cell_count = NULL;
  cs_int_t  *cell_cells_idx = NULL, *cell_cells_lst = NULL;

  const cs_int_t  n_cells = mesh->n_cells;
  const cs_int_t  n_cells_wghosts = mesh->n_cells_with_ghosts;
  const cs_int_t  *face_cells = mesh->i_face_cells;
  const cs_int_t  *fac_vtx_idx = mesh->i_face_vtx_idx;
  const cs_int_t  *fac_vtx_lst = mesh->i_face_vtx_lst;

  /* Allocate and initialize buffers */

  BFT_MALLOC(cell_cells_idx, n_cells + 1, cs_int_t);
  BFT_MALLOC(cell_buffer, n_cells_wghosts + n_cells, cs_int_t);

  cell_tag = &(cell_buffer[0]);
  cell_count = &(cell_buffer[n_cells_wghosts]);

  cell_cells_idx[0] = 1;
  for (i = 0; i < n_cells; i++) {
    cell_cells_idx[i+1] = 0;
    cell_count[i] = 0;
  }

  for (i = 0; i < n_cells_wghosts; i++)
    cell_tag[i] = -1;

  /* Define index */

  for (i_cel = 0; i_cel < n_cells; i_cel++) {

    /* First loop on faces to tag cells sharing a face */

    for (i = cell_i_faces_idx[i_cel]; i < cell_i_faces_idx[i_cel+1]; i++) {

      fac_id = CS_ABS(cell_i_faces_lst[i-1]) - 1;

      cell_tag[face_cells[2*fac_id] - 1] = i_cel;
      cell_tag[face_cells[2*fac_id+1] - 1] = i_cel;

    } /* End of loop on cell's faces */

    /* Second loop on faces to update index */

    for (i = cell_i_faces_idx[i_cel]; i < cell_i_faces_idx[i_cel+1]; i++) {

      fac_id = CS_ABS(cell_i_faces_lst[i-1]) - 1;

      for (i_vtx = fac_vtx_idx[fac_id];
           i_vtx < fac_vtx_idx[fac_id+1]; i_vtx++) {

        vtx_id = fac_vtx_lst[i_vtx-1] - 1;

        /* For cells belonging to this rank, get vertex -> cells connect. */

        for (j = vtx_cells_idx[vtx_id]; j < vtx_cells_idx[vtx_id+1]; j++) {

          cell_id = vtx_cells_lst[j];

          if (cell_tag[cell_id] != i_cel) {
            cell_cells_idx[i_cel+1] += 1;
            cell_tag[cell_id] = i_cel;
          }

        }

        if (n_cells_wghosts - n_cells > 0) { /* If there are ghost cells */

          /* For ghost cells, get vertex -> ghost cells connect. */

          for (j = vtx_gcells_idx[vtx_id]; j < vtx_gcells_idx[vtx_id+1]; j++) {

            cell_id = vtx_gcells_lst[j] + n_cells;

            if (cell_tag[cell_id] != i_cel) {
              cell_cells_idx[i_cel+1] += 1;
              cell_tag[cell_id] = i_cel;
            }

          }

        } /* If there are ghost cells */

      } /* End of loop on vertices */

    } /* End of loop on cell's faces */

  } /* End of loop on cells */

  /* Create index */

  for (i = 0; i < n_cells; i++)
    cell_cells_idx[i+1] += cell_cells_idx[i];

  for (i = 0; i < n_cells_wghosts; i++)
    cell_tag[i] = -1;

  BFT_MALLOC(cell_cells_lst, cell_cells_idx[n_cells] - 1, cs_int_t);

  /* Fill list */

  for (i_cel = 0; i_cel < n_cells; i_cel++) {

    /* First loop on faces to tag cells sharing a face */

    for (i = cell_i_faces_idx[i_cel]; i < cell_i_faces_idx[i_cel+1]; i++) {

      fac_id = CS_ABS(cell_i_faces_lst[i-1]) - 1;

      cell_tag[face_cells[2*fac_id] - 1] = i_cel;
      cell_tag[face_cells[2*fac_id+1] - 1] = i_cel;

    } /* End of loop on cell's faces */

    /* Second loop on faces to update list */

    for (i = cell_i_faces_idx[i_cel]; i < cell_i_faces_idx[i_cel+1]; i++) {

      fac_id = CS_ABS(cell_i_faces_lst[i-1]) - 1;

      for (i_vtx = fac_vtx_idx[fac_id];
           i_vtx < fac_vtx_idx[fac_id+1]; i_vtx++) {

        vtx_id = fac_vtx_lst[i_vtx-1] - 1;

        /* For cells belonging to this rank, get vertex -> cells connect. */

        for (j = vtx_cells_idx[vtx_id]; j < vtx_cells_idx[vtx_id+1]; j++) {

          cell_id = vtx_cells_lst[j];

          if (cell_tag[cell_id] != i_cel) {

            shift = cell_cells_idx[i_cel] - 1 + cell_count[i_cel];
            cell_cells_lst[shift] = cell_id + 1;
            cell_tag[cell_id] = i_cel;
            cell_count[i_cel] += 1;

          } /* Add this cell_id */

        }

        if (n_cells_wghosts - n_cells > 0) { /* If there are ghost cells */

          /* For ghost cells, get vertex -> ghost cells connect. */

          for (j = vtx_gcells_idx[vtx_id]; j < vtx_gcells_idx[vtx_id+1]; j++){

            cell_id = vtx_gcells_lst[j] + n_cells;

            if (cell_tag[cell_id] != i_cel) {

              shift = cell_cells_idx[i_cel]  - 1 + cell_count[i_cel];
              cell_cells_lst[shift] = cell_id + 1;
              cell_tag[cell_id] = i_cel;
              cell_count[i_cel] += 1;

            }

          }

        } /* If there are ghost cells */

      } /* End of loop on vertices */

    } /* End of loop on cell's faces */

  } /* End of loop on cells */

  *p_cell_cells_idx = cell_cells_idx;
  *p_cell_cells_lst = cell_cells_lst;

  /* Free memory */

  BFT_FREE(cell_buffer);

}

/*============================================================================
 * Public function definitions for Fortran API
 *============================================================================*/

/*----------------------------------------------------------------------------
 * Define a new "cell -> cells" connectivity for the  extended neighborhood
 * in case of computation of gradient whith the least squares algorithm
 * (imrgra = 3).
 * The "cell -> cells" connectivity is clipped by a non-orthogonality
 * criterion.
 *
 * Warning   :  Only cells sharing a vertex or vertices
 *              (not a face => mesh->face_cells) belong to the
 *              "cell -> cells" connectivity.
 *
 * Fortran Interface :
 *
 * SUBROUTINE REDVSE
 * *****************
 *    & ( ANOMAX )
 *
 * parameters:
 *   anomax  -->  non-orthogonality angle (rad) above which cells
 *                are selected for the extended neighborhood
 *----------------------------------------------------------------------------*/

void
CS_PROCF (redvse, REDVSE) (const cs_real_t  *anomax)
{
  cs_int_t  i, face_id, cell_id, cell_i, cell_j;
  size_t  init_cell_cells_connect_size;

  cs_real_t  v_ij[3];
  cs_real_t  face_normal[3];
  cs_real_t  norm_ij, face_norm, cos_ij_fn;
  cs_real_t  dprod;
  double     ratio;

  cs_int_t  n_deleted_cells = 0;
  cs_int_t  previous_idx = 0, new_idx = -1;

  cs_mesh_t  *mesh = cs_glob_mesh;
  cs_mesh_quantities_t  *mesh_quantities = cs_glob_mesh_quantities;
  cs_int_t  *vtx_cells_idx = NULL, *vtx_cells_lst = NULL;
  cs_int_t  *cell_cells_idx = mesh->cell_cells_idx;
  cs_int_t  *cell_cells_lst = mesh->cell_cells_lst;

  const cs_int_t  n_faces = mesh->n_i_faces;
  const cs_int_t  n_cells = mesh->n_cells;
  const cs_int_t  *face_cells = mesh->i_face_cells;

  const cs_real_t  cos_ij_fn_min = cos((*anomax));
  const cs_real_t  *cell_cen = mesh_quantities->cell_cen;

  /* Currently limited to 1 call, but the algorithm would work just the same
     with multiple calls (as we re-build a new cell -> cells connectivity
     instead of just filtering the one we already have) */

  static  cs_int_t _first_call = 0;

  enum {X, Y, Z};

#define CS_LOC_MODULE(vect) \
     sqrt(vect[X] * vect[X] + vect[Y] * vect[Y] + vect[Z] * vect[Z])

  assert(mesh->dim == 3);

  /* First call: select the cells */

  if (_first_call == 0) {

    /* Update counter */

    _first_call = 1;

    /* Warn if there is no extended neighborhood */

    if (   mesh->cell_cells_lst == NULL
        || mesh->cell_cells_idx == NULL
        || mesh->halo_type == CS_HALO_STANDARD) {
      bft_printf
        (_("\n"
           "WARNING\n"
           "The extended neighborhood is empty whereas the least-squares\n"
           "method on extended neighborhood for gradient computation\n"
           "is activated. This situation can arise in some particular\n"
           "cases (1D mesh). Verify that it is your case, otherwise\n"
           "contact support.\n"));
    }
    else {


    /*
      For each internal face, we select in the extended neighborhood
      of the two cells sharing this face all the cells sharing a
      vertex of this face if the non-orthogonality angle is above a
      criterion.
    */

    /*
      First: re-build a "vertex -> cells" connectivity
      ------------------------------------------------
      We have to invert the "face -> vertices" connectivity and then
      we will use the "face -> cells" conectivity.
    */

    _create_vtx_cells_connect(mesh,
                              &vtx_cells_idx,
                              &vtx_cells_lst);

    /* Tag cells to eliminate (set a negative number) */

    for (face_id = 0 ; face_id < n_faces ; face_id++) {

      /* We compute the cosine of the non-orthogonality angle
         of internal faces (angle between the normal of the face
         and the line between I (center of the cell I) and J (center
         of the cell J) */

      /* Vector IJ and normal of the face */

      cell_i = face_cells[2*face_id] - 1;
      cell_j = face_cells[2*face_id + 1] - 1;
      dprod = 0;

      for (i = 0; i < 3; i++) {
        v_ij[i] = cell_cen[3*cell_j + i] - cell_cen[3*cell_i + i];
        face_normal[i] = mesh_quantities->i_face_normal[3*face_id + i];
        dprod += v_ij[i]*face_normal[i];
      }

      norm_ij = CS_LOC_MODULE(v_ij);
      face_norm = CS_LOC_MODULE(face_normal);

      assert(norm_ij > 0.);
      assert(face_norm > 0.);

      /* Dot product : norm_ij . face_norm */

      cos_ij_fn = dprod / (norm_ij * face_norm);

      /* Comparison to a predefined limit.
         This is non-orthogonal if we are below the limit and so we keep
         the cell in the extended neighborhood of the two cells sharing
         the face. (The cell is tagged (<0) then we will change the sign
         and eliminate all cells < 0 */

      if (cos_ij_fn <= cos_ij_fn_min) {

        /* For each cell sharing the face : intersection between
           cells in the extended neighborhood and cells sharing a
           vertex of the face. */

        _tag_cells(face_id, cell_i, mesh, vtx_cells_idx, vtx_cells_lst);
        _tag_cells(face_id, cell_j, mesh, vtx_cells_idx, vtx_cells_lst);

      }

    } /* End of loop on faces */

    /* Free "vertex -> cells" connectivity */

    BFT_FREE(vtx_cells_idx);
    BFT_FREE(vtx_cells_lst);

    /* Change all signs in cell_cells_lst in order to have cells to
       eliminate < 0 */

    for (i = 0 ; i < mesh->cell_cells_idx[n_cells]-1 ; i++)
      mesh->cell_cells_lst[i] = - mesh->cell_cells_lst[i];

    /* Delete negative cells */

    init_cell_cells_connect_size = cell_cells_idx[n_cells] - 1;

    for (cell_id = 0; cell_id < n_cells; cell_id++) {

      for (i = previous_idx; i < cell_cells_idx[cell_id+1] - 1; i++) {

        if (cell_cells_lst[i] > 0) {
          new_idx++;
          cell_cells_lst[new_idx] = cell_cells_lst[i];
        }
        else
          n_deleted_cells++;

      } /* End of loop on cells in the extended neighborhood of cell_id+1 */

      previous_idx = cell_cells_idx[cell_id+1] - 1;
      cell_cells_idx[cell_id+1] -= n_deleted_cells;

    } /* End of loop on cells */

    /* Reallocation of cell_cells_lst */

    BFT_REALLOC(mesh->cell_cells_lst, cell_cells_idx[n_cells]-1, cs_int_t);

    /* Output for listing */

#if defined(_CS_HAVE_MPI)

    if (cs_glob_base_nbr > 1) {

      unsigned long count_g[2];
      unsigned long count_l[2] = {init_cell_cells_connect_size,
                                  n_deleted_cells};
      MPI_Allreduce(count_l, count_g, 2, MPI_UNSIGNED_LONG,
                    MPI_SUM, cs_glob_base_mpi_comm);

      init_cell_cells_connect_size = count_g[0];
      n_deleted_cells = count_g[1];

    }

#endif

    ratio = 100. * (init_cell_cells_connect_size - n_deleted_cells)
                 / init_cell_cells_connect_size;

    bft_printf
      (_("\n"
         " Extended neighborhood reduced by non-orthogonality\n"
         " --------------------------------------------------\n"
         "\n"
         " Size of complete cell-cell connectivity: %12lu\n"
         " Size of filtered cell-cell conectivity:  %12lu\n"
         " %lu cells removed, for a ratio of %4.2g %% used\n"),
       (unsigned long)init_cell_cells_connect_size,
       (unsigned long)(init_cell_cells_connect_size - n_deleted_cells),
       (unsigned long)n_deleted_cells,
       ratio);

#if 0 /* For debugging purpose */
      for (i = 0; i < mesh->n_cells ; i++) {
        cs_int_t  j;
        bft_printf(" cell %d :: ", i+1);
        for (j = mesh->cell_cells_idx[i]-1;
             j < mesh->cell_cells_idx[i+1]-1;
             j++)
          bft_printf(" %d ", mesh->cell_cells_lst[j]);
        bft_printf("\n");
      }
#endif

    } /* If there is extended neighborhood */

  } /* If _first_call == 0 */

}

/*----------------------------------------------------------------------------
 * Compute filters for dynamic models. This function deals with the standard
 * or extended neighborhood.
 *
 * Fortran Interface :
 *
 * SUBROUTINE CFILTR (VAR, F_VAR, WBUF1, WBUF2)
 * *****************
 *
 * DOUBLE PRECISION(*) var[]      --> array of variables to filter
 * DOUBLE PRECISION(*) f_var[]    --> filtered variable array
 * DOUBLE PRECISION(*) wbuf1[]    --> working buffer
 * DOUBLE PRECISION(*) wbuf2[]    --> working buffer
 *----------------------------------------------------------------------------*/

void
CS_PROCF (cfiltr, CFILTR)(cs_real_t    var[],
                          cs_real_t    f_var[],
                          cs_real_t    wbuf1[],
                          cs_real_t    wbuf2[])
{
  cs_int_t  i, j, k;

  const cs_mesh_t  *mesh = cs_glob_mesh;
  const cs_int_t  n_cells = mesh->n_cells;
  const cs_int_t  *cell_cells_idx = mesh->cell_cells_idx;
  const cs_int_t  *cell_cells_lst = mesh->cell_cells_lst;
  const cs_real_t  *cell_vol = cs_glob_mesh_quantities->cell_vol;

  /* Synchronize variable */

  if (mesh->halo != NULL) {

    cs_halo_sync_var(mesh->halo, CS_HALO_EXTENDED, var);

    if (mesh->n_init_perio > 0)
      cs_perio_sync_var_scal(mesh->halo,
                             CS_HALO_EXTENDED,
                             CS_PERIO_ROTA_COPY,
                             var);

  }

  /* Allocate and initialize working buffers */

  for (i = 0; i < n_cells; i++) {
    wbuf1[i] = 0;
    wbuf2[i] = 0;
  }

  /* Define filtered variable array */

  for (i = 0; i < n_cells; i++) {

    wbuf1[i] += var[i] * cell_vol[i];
    wbuf2[i] += cell_vol[i];

    /* Loop on connected cells (without cells sharing a face) */

    for (j = cell_cells_idx[i] - 1; j < cell_cells_idx[i+1] - 1; j++) {

      k = cell_cells_lst[j] - 1;
      wbuf1[i] += var[k] * cell_vol[k];
      wbuf2[i] += cell_vol[k];

    }

  } /* End of loop on cells */

  for (k = 0; k < mesh->n_i_faces; k++) {

    i = mesh->i_face_cells[2*k] - 1;
    j = mesh->i_face_cells[2*k + 1] - 1;

    wbuf1[i] += var[j] * cell_vol[j];
    wbuf2[i] += cell_vol[j];
    wbuf1[j] += var[i] * cell_vol[i];
    wbuf2[j] += cell_vol[i];

  }

  for (i = 0; i < n_cells; i++)
    f_var[i] = wbuf1[i]/wbuf2[i];

  /* Synchronize variable */

  if (mesh->halo != NULL) {

    cs_halo_sync_var(mesh->halo, CS_HALO_STANDARD, f_var);

    if (mesh->n_init_perio > 0)
      cs_perio_sync_var_scal(mesh->halo,
                             CS_HALO_STANDARD,
                             CS_PERIO_ROTA_COPY,
                             f_var);

  }
}

/*----------------------------------------------------------------------------
 * Create the  "cell -> cells" connectivity
 *
 * parameters:
 *   mesh           <->  pointer to a mesh structure.
 *   gcell_vtx_idx  <--  pointer to the connectivity index
 *   gcell_vtx_lst  <--  pointer to the connectivity list
 *---------------------------------------------------------------------------*/

void
cs_ext_neighborhood_define(cs_mesh_t   *mesh,
                           cs_int_t    *gcell_vtx_idx,
                           cs_int_t    *gcell_vtx_lst)
{
  cs_int_t  *vtx_gcells_idx = NULL, *vtx_gcells_lst = NULL;
  cs_int_t  *vtx_cells_idx = NULL, *vtx_cells_lst = NULL;
  cs_int_t  *cell_i_faces_idx = NULL, *cell_i_faces_lst = NULL;
  cs_int_t  *cell_cells_idx = NULL, *cell_cells_lst = NULL;

  cs_halo_t  *halo = mesh->halo;

  /* Get "cell -> faces" connectivity for the local mesh */

  _get_cell_i_faces_connectivity(mesh,
                                 &cell_i_faces_idx,
                                 &cell_i_faces_lst);

  /* Create a "vertex -> cell" connectivity */

  _create_vtx_cells_connect2(mesh,
                             cell_i_faces_idx,
                             cell_i_faces_lst,
                             &vtx_cells_idx,
                             &vtx_cells_lst);

  if (cs_mesh_n_g_ghost_cells(mesh) > 0) {

    /* Create a "vertex -> ghost cells" connectivity */

    _create_vtx_gcells_connect(halo,
                               mesh->n_vertices,
                               gcell_vtx_idx,
                               gcell_vtx_lst,
                               &vtx_gcells_idx,
                               &vtx_gcells_lst);

    mesh->vtx_gcells_idx = vtx_gcells_idx;
    mesh->vtx_gcells_lst = vtx_gcells_lst;

  }

  /* Create the "cell -> cells" connectivity for the extended halo */

  _create_cell_cells_connect(mesh,
                             cell_i_faces_idx,
                             cell_i_faces_lst,
                             vtx_gcells_idx,
                             vtx_gcells_lst,
                             vtx_cells_idx,
                             vtx_cells_lst,
                             &cell_cells_idx,
                             &cell_cells_lst);

  mesh->cell_cells_idx = cell_cells_idx;
  mesh->cell_cells_lst = cell_cells_lst;

  /* Free memory */

  BFT_FREE(cell_i_faces_idx);
  BFT_FREE(cell_i_faces_lst);
  BFT_FREE(vtx_cells_idx);
  BFT_FREE(vtx_cells_lst);

}

/*----------------------------------------------------------------------------*/

END_C_DECLS

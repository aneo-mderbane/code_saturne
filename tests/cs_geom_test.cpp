/*============================================================================
 * Unit test for some geometrical algorithms.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <bft_error.h>
#include <bft_mem.h>
#include <bft_printf.h>

#include "cs_math.h"

/*---------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
 * Function that performs the intersection between a plane and a polygon.
 * It returns the resulting polygon at the "inner" side of the plane
 * according to its normal.
 *
 * parameters:
 *   nb_vertex     <--> number of vertices of the polygon
 *   vertex_coords <--> coordinates of the vertices (size : 3*nb_vertex)
 *   plane         <--  plane definition (point + unit normal)
 ----------------------------------------------------------------------------*/

static void
_polygon_plane_intersection(int           *nb_vertex,
                            cs_real_3_t  **vertex_coord,
                            cs_real_t      plane[6])
{
  /* Initial number of vertices in the polygon */
  cs_lnum_t n_vtx = *nb_vertex;
  cs_real_3_t *vtx = *vertex_coord;

  cs_real_3_t *new_vtx = NULL;
  BFT_MALLOC(new_vtx, n_vtx + 1, cs_real_3_t);
  int j = 0;

  cs_real_t tolerance_factor = 0.1; /* tunable in "real" code */

  /* Now we check which edge is intersected by the plane */
  for (cs_lnum_t i = 0; i < n_vtx; i++) {
    /* In each loop iteration we check if [v1, v2] intersects the plane
       and if v2 belongs to the negative half space */
    cs_lnum_t v0 = (i-1+n_vtx) % n_vtx;
    cs_lnum_t v1 = i;
    cs_lnum_t v2 = (i+1) % n_vtx;
    cs_lnum_t v3 = (i+2) % n_vtx;

    cs_real_t tolerance_v1 =   tolerance_factor
                             * cs_math_fmin(cs_math_3_distance(vtx[v0], vtx[v1]),
                                            cs_math_3_distance(vtx[v1], vtx[v2]));
    cs_real_t tolerance_v2 =   tolerance_factor
                             * cs_math_fmin(cs_math_3_distance(vtx[v1], vtx[v2]),
                                            cs_math_3_distance(vtx[v2], vtx[v3]));

    cs_real_t xn1 = cs_math_3_distance_dot_product(vtx[v1], plane, plane+3);
    cs_real_t xn2 = cs_math_3_distance_dot_product(vtx[v2], plane, plane+3);

    /* If [v1, v2] (almost) tangent then add v2 projected on the plane */
    if (cs_math_fabs(xn1) <= tolerance_v1 && (cs_math_fabs(xn2) <= tolerance_v2)) {
      assert(j <= n_vtx);
      for (cs_lnum_t dir = 0; dir < 3; dir++)
        new_vtx[j][dir] = vtx[v2][dir] + xn2 * plane[dir+3];
      j++;
    }

    else {
      /* If intersection and if its not close to v1 or v2 then new vertex */
      if (xn1*xn2 < 0) {

        /* Compute the parametric coordinate t (should always be well defined) */
        cs_real_t xd = cs_math_3_distance_dot_product(vtx[v1], vtx[v2], plane+3);
        cs_real_t t = xn1/xd;
        cs_real_t edge_length = cs_math_3_distance(vtx[v1], vtx[v2]);
        cs_real_t d1 = t * edge_length, d2 = (1 - t) * edge_length;

        if (d1 > tolerance_v1 && d2 > tolerance_v2) {
          assert(j <= n_vtx);
          for (cs_lnum_t dir = 0; dir < 3; dir++)
            new_vtx[j][dir] = vtx[v1][dir] + t * (vtx[v2][dir] - vtx[v1][dir]);
          j++;
        }
      }

      /* If v2 inside the plane (with tolerance) then add v2, if its close project
         it on to the plane */
      if (xn2 >= -tolerance_v2) {
        assert(j <= n_vtx);
        bool v2_close = cs_math_fabs(xn2) < tolerance_v2;
        for (cs_lnum_t dir = 0; dir < 3; dir++)
          new_vtx[j][dir] = vtx[v2][dir] + v2_close * xn2 * plane[dir+3];
        j++;
      }
    }
  }

  BFT_REALLOC(vtx, j, cs_real_3_t);

  for (cs_lnum_t i = 0; i < j; i++) {
    printf("%d: ", i);
    for (cs_lnum_t dir = 0; dir < 3; dir ++) {
      vtx[i][dir] = new_vtx[i][dir];
      printf(" %g", vtx[i][dir]);
    }
    printf("\n");
  }

  BFT_FREE(new_vtx);

  *nb_vertex = j;
  *vertex_coord = vtx;
}

/*---------------------------------------------------------------------------*/

int
main (int argc, char *argv[])
{
  CS_UNUSED(argc);
  CS_UNUSED(argv);

  int nb_vertex = 4;
  cs_real_3_t *vertex_coord;
  BFT_MALLOC(vertex_coord, 4, cs_real_3_t);

  vertex_coord[0][0] = 9;
  vertex_coord[0][1] = 1;
  vertex_coord[0][2] = 0;

  vertex_coord[1][0] = 10;
  vertex_coord[1][1] = 1;
  vertex_coord[1][2] = 0;

  vertex_coord[2][0] = 10;
  vertex_coord[2][1] = 2;
  vertex_coord[2][2] = 0;

  vertex_coord[3][0] = 9;
  vertex_coord[3][1] = 2;
  vertex_coord[3][2] = 1e-13;

  cs_real_t plane[6] = {10, 2, 0,
                        0, 0, 1};

  _polygon_plane_intersection(&nb_vertex,
                              &vertex_coord,
                              plane);

  exit(EXIT_SUCCESS);
}

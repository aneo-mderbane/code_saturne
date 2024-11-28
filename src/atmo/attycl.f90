!-------------------------------------------------------------------------------

! This file is part of code_saturne, a general-purpose CFD tool.
!
! Copyright (C) 1998-2024 EDF S.A.
!
! This program is free software; you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the Free Software
! Foundation; either version 2 of the License, or (at your option) any later
! version.
!
! This program is distributed in the hope that it will be useful, but WITHOUT
! ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
! FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
! Street, Fifth Floor, Boston, MA 02110-1301, USA.

!-------------------------------------------------------------------------------
!> \file attycl.f90
!> \brief Automatic boundary conditions for atmospheric module
!>   (based on meteo file)

!> \brief Automatically compute the boundary conditions from the meteo file
!>        or from the imbrication profiles
!-------------------------------------------------------------------------------
! Arguments
!______________________________________________________________________________.
!  mode           name          role                                           !
!______________________________________________________________________________!
!> \param[in]   itypfb          boundary face types
!> \param[out]  icodcl          face boundary condition code
!>                               - 1 Dirichlet
!>                               - 2 Radiative outlet
!>                               - 3 Neumann
!>                               - 4 sliding and
!>                                 \f$ \vect{u} \cdot \vect{n} = 0 \f$
!>                               - 5 smooth wall and
!>                                 \f$ \vect{u} \cdot \vect{n} = 0 \f$
!>                               - 6 rough wall and
!>                                 \f$ \vect{u} \cdot \vect{n} = 0 \f$
!>                               - 9 free inlet/outlet
!>                                 (input mass flux blocked to 0)
!>                               - 13 Dirichlet for the advection operator and
!>                                    Neumann for the diffusion operator
!> \param[out]  rcodcl          Boundary conditions value
!>                               - rcodcl(1) value of the dirichlet
!>                               - rcodcl(2) value of the exterior exchange
!>                                 coefficient (infinite if no exchange)
!>                               - rcodcl(3) value flux density
!>                                 (negative if gain) in w/m2
!>                                 -# for the velocity \f$ (\mu+\mu_T)
!>                                    \gradv \vect{u} \cdot \vect{n}  \f$
!>                                 -# for the pressure \f$ \Delta t
!>                                    \grad P \cdot \vect{n}  \f$
!>                                 -# for a scalar \f$ cp \left( K +
!>                                     \dfrac{K_T}{\sigma_T} \right)
!>                                     \grad T \cdot \vect{n} \f$
!
!-------------------------------------------------------------------------------
subroutine attycl ( itypfb, icodcl, rcodcl )

!===============================================================================
! Module files
!===============================================================================

use paramx
use numvar
use optcal
use cstphy
use cstnum
use dimens, only: nvar
use entsor
use parall
use ppppar
use ppthch
use ppincl
use mesh
use field
use atincl
use atchem
use atimbr
use sshaerosol
use cs_c_bindings

!===============================================================================

implicit none

! Arguments

procedure() :: mscrss

integer          itypfb(nfabor)
integer          icodcl(nfabor,nvar)
double precision rcodcl(nfabor,nvar,3)

! Local variables

integer          ifac, iel, ilelt
integer          ii, nbrsol, nelts
integer          jsp, isc, ivar
integer          fid_axz, fid_mrij
double precision zent, vs, xwent, dnorm_vel
double precision vel_dir(3)
double precision rij_loc(6)
double precision tpent, qvent,ncent
double precision xcent
double precision pp, dum

integer, dimension(:), pointer :: elt_ids

double precision, dimension(:), pointer :: coefap
double precision, pointer, dimension(:)   :: bvar_tempp
double precision, pointer, dimension(:)   :: bvar_total_water

!===============================================================================
! 1.  INITIALISATIONS
!===============================================================================


! Soil atmosphere boundary conditions
!------------------------------------
if (iatsoil.ge.1) then
  call field_get_val_s_by_name("soil_pot_temperature", bvar_tempp)
  call field_get_val_s_by_name("soil_total_water", bvar_total_water)
  call atmo_get_soil_zone(nelts, nbrsol, elt_ids)

  do ilelt = 1, nelts

    ifac = elt_ids(ilelt) + 1 ! C > Fortran

    ! Rough wall if no specified
    ! Note: roughness and thermal roughness are computed in solmoy
    if (itypfb(ifac).eq.0) itypfb(ifac) = iparug

    if (iscalt.ne.-1) then
      ! If not yet specified
      if (rcodcl(ifac,isca(iscalt),1).gt.rinfin*0.5d0)  then
        ! Dirichlet with wall function Expressed directly in term of
        ! potential temperature
        icodcl(ifac,isca(iscalt))   = -6
        rcodcl(ifac,isca(iscalt),1) = bvar_tempp(ilelt)
      endif
    endif
    if (ippmod(iatmos).eq.2) then
      ! If not yet specified
      if (rcodcl(ifac,isca(iymw),1).gt.rinfin*0.5d0)  then
        icodcl(ifac, isca(iymw)) = 6
        rcodcl(ifac, isca(iymw),1) = bvar_total_water(ilelt)
      endif
    endif

  enddo
endif

!==============================================================================
! Imbrication
!==============================================================================

if (imbrication_flag) then

  call summon_cressman(ttcabs)

  if (cressman_u) then
    call mscrss(id_u,2,rcodcl(1,iu,1))
    if (imbrication_verbose) then
      do ifac = 1, max(nfabor,100), 1
        write(nfecra,*)"attycl::xbord,ybord,zbord,ubord=", cdgfbo(1,ifac), &
             cdgfbo(2,ifac),                                               &
             cdgfbo(3,ifac),                                               &
             rcodcl(ifac, iu, 1)
      enddo
    endif
  endif

  if (cressman_v) then
    call mscrss(id_v,2,rcodcl(1,iv,1))
    if(imbrication_verbose)then
      do ifac = 1, max(nfabor,100), 1
        write(nfecra,*)"attycl::xbord,ybord,zbord,vbord=", cdgfbo(1,ifac), &
             cdgfbo(2,ifac),                                               &
             cdgfbo(3,ifac),                                               &
             rcodcl(ifac, iv, 1)
      enddo
    endif
  endif

  if (cressman_tke) then
    call mscrss(id_tke,2,rcodcl(1,ik,1))
    if(imbrication_verbose)then
      do ifac = 1, max(nfabor,100), 1
        write(nfecra,*)"attycl::xbord,ybord,zbord,tkebord=",cdgfbo(1,ifac), &
             cdgfbo(2,ifac),                                                &
             cdgfbo(3,ifac),                                                &
             rcodcl(ifac, ik, 1)
      enddo
    endif
  endif

  if (cressman_eps) then
    call mscrss(id_eps,2,rcodcl(1,iep,1))
    if(imbrication_verbose)then
      do ifac = 1, max(nfabor,100), 1
        write(nfecra,*)"attycl::xbord,ybord,zbord,epsbord=",cdgfbo(1,ifac), &
             cdgfbo(2,ifac),                                                &
             cdgfbo(3,ifac),                                                &
             rcodcl(ifac,iep,1)
      enddo
    endif
  endif

  if (cressman_theta .and. ippmod(iatmos).ge.1) then
    call mscrss(id_theta, 2, rcodcl(1,isca(iscalt),1))
    if (imbrication_verbose) then
      do ifac = 1, max(nfabor,100), 1
        write(nfecra,*)"attycl::xbord,ybord,zbord,thetabord=",cdgfbo(1,ifac), &
             cdgfbo(2,ifac),                                                  &
             cdgfbo(3,ifac),                                                  &
             rcodcl(ifac,isca(iscalt),1)
      enddo
    endif
  endif

  if (cressman_qw .and. ippmod(iatmos).ge.2) then
    call mscrss(id_qw, 2, rcodcl(1,isca(iymw),1))
    if (imbrication_verbose) then
      do ifac = 1, max(nfabor,100), 1
        write(nfecra,*)"attycl::xbord,ybord,zbord,qwbord=",cdgfbo(1,ifac), &
             cdgfbo(2,ifac),                                               &
             cdgfbo(3,ifac),                                               &
             rcodcl(ifac,isca(iymw),1)
      enddo
    endif
  endif

  if (cressman_nc .and. ippmod(iatmos).ge.2) then
    call mscrss(id_nc, 2, rcodcl(1,isca(intdrp),1))
    if (imbrication_verbose) then
      do ifac = 1, max(nfabor,100), 1
        write(nfecra,*)"attycl::xbord,ybord,zbord,ncbord=",cdgfbo(1,ifac), &
             cdgfbo(2,ifac),                                               &
             cdgfbo(3,ifac),                                               &
             rcodcl(ifac,isca(intdrp),1)
      enddo
    endif
  endif

endif ! imbrication_flag

! Atmospheric gaseous chemistry
if (ichemistry.ge.1) then

  do ifac = 1, nfabor

    if (itypfb(ifac).eq.ientre) then

      zent = cdgfbo(3,ifac)

      ! For species present in the concentration profiles chemistry file,
      ! profiles are used here as boundary conditions if boundary conditions have
      ! not been treated earlier (eg, in cs_user_boundary_conditions)
      do ii = 1, nespgi
        if (rcodcl(ifac,isca(isca_chem(idespgi(ii))),1).gt.0.5d0*rinfin) then
          call intprf &
            (nbchmz, nbchim,                                               &
            zproc, tchem, espnum, zent  , ttcabs, xcent )
          ! The first nespg user scalars are supposed to be chemical species
          rcodcl(ifac,isca(isca_chem(idespgi(ii))),1) = xcent
        endif
      enddo

      ! For other species zero Dirichlet conditions are imposed,
      ! unless they have already been treated earlier (eg, in cs_user_boundary_conditions)
      do ii =1 , nespg
        if (rcodcl(ifac,isca(isca_chem(ii)),1).gt.0.5d0*rinfin) then
          rcodcl(ifac,isca(isca_chem(ii)),1) = 0.0d0
        endif
      enddo

    endif

  enddo

endif

! Atmospheric aerosol chemistry
if (iaerosol.ne.CS_ATMO_AEROSOL_OFF) then

  do ifac = 1, nfabor

    if (itypfb(ifac).eq.ientre) then

      do jsp = 1, nlayer_aer*n_aer+n_aer
        isc = isca_chem(nespg + jsp)
        if (rcodcl(ifac,isca(isc),1).gt.0.5d0*rinfin) &
            rcodcl(ifac,isca(isc),1) = dlconc0(jsp)
      enddo

    ! For other species zero dirichlet conditions are imposed,
    ! unless they have already been treated earlier (eg, in usatcl)
      do ii = 1, nlayer_aer*n_aer+n_aer
        isc = isca_chem(nespg + ii)
        if (rcodcl(ifac,isca(isc),1).gt.0.5d0*rinfin) &
            rcodcl(ifac,isca(isc),1) = 0.0d0
      enddo

      ! For gaseous species which have not been treated earlier
      ! (for example species not present in the third gaseous scheme,
      ! which can be treated in usatcl of with the file chemistry)
      ! zero dirichlet conditions are imposed
      do ii = 1, nespg
        isc = isca_chem(ii)
        if (rcodcl(ifac,isca(isc),1).gt.0.5d0*rinfin) &
          rcodcl(ifac,isca(isc),1) = 0.0d0
      enddo

    endif

  enddo

endif

!--------
! Formats
!--------


!----
! End
!----

return
end subroutine attycl

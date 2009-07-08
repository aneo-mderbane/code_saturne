!-------------------------------------------------------------------------------

!     This file is part of the Code_Saturne Kernel, element of the
!     Code_Saturne CFD tool.

!     Copyright (C) 1998-2008 EDF S.A., France

!     contact: saturne-support@edf.fr

!     The Code_Saturne Kernel is free software; you can redistribute it
!     and/or modify it under the terms of the GNU General Public License
!     as published by the Free Software Foundation; either version 2 of
!     the License, or (at your option) any later version.

!     The Code_Saturne Kernel is distributed in the hope that it will be
!     useful, but WITHOUT ANY WARRANTY; without even the implied warranty
!     of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!     GNU General Public License for more details.

!     You should have received a copy of the GNU General Public License
!     along with the Code_Saturne Kernel; if not, write to the
!     Free Software Foundation, Inc.,
!     51 Franklin St, Fifth Floor,
!     Boston, MA  02110-1301  USA

!-------------------------------------------------------------------------------

                  subroutine cpvarp
!================


!===============================================================================
!  FONCTION  :
!  ---------

!       INIT DES POSITIONS DES VARIABLES TRANSPORTEES POUR
!                COMBUSTION CHARBON PULVERISE

!-------------------------------------------------------------------------------
! Arguments
!__________________.____._____.________________________________________________.
!    nom           !type!mode !                   role                         !
!__________________!____!_____!________________________________________________!
!__________________!____!_____!________________________________________________!

!     TYPE : E (ENTIER), R (REEL), A (ALPHANUMERIQUE), T (TABLEAU)
!            L (LOGIQUE)   .. ET TYPES COMPOSES (EX : TR TABLEAU REEL)
!     MODE : <-- donnee, --> resultat, <-> Donnee modifiee
!            --- tableau de travail
!===============================================================================

implicit none

!===============================================================================
!     DONNEES EN COMMON
!===============================================================================

include "paramx.h"
include "dimens.h"
include "numvar.h"
include "optcal.h"
include "cstphy.h"
include "entsor.h"
include "cstnum.h"
include "ppppar.h"
include "ppthch.h"
include "coincl.h"
include "cpincl.h"
include "ppincl.h"
include "ppcpfu.h"
include "ihmpre.h"

!===============================================================================

integer        icla,  is, iphas, icha, isc , is1

!===============================================================================
!===============================================================================
! 1. DEFINITION DES POINTEURS
!===============================================================================

! ---> Variables propres a la suspension gaz - particules

ihm   = iscapp(1)

! ---> Variables propres a la phase dispersee

do icla = 1, nclacp
  is = 1+icla
  inp(icla) = iscapp(is)
  is = 1+1*nclacp+icla
  ixch(icla)= iscapp(is)
  is = 1+2*nclacp+icla
  ixck(icla) = iscapp(is)
  if ( ippmod(icp3pl) .eq. 1 ) then
    is = 1+3*nclacp+icla
    ixwt(icla) = iscapp(is)

    is = 1+4*nclacp+icla
    ih2(icla) = iscapp(is)

  else
    is = 1+3*nclacp+icla
    ih2(icla) = iscapp(is)
  endif

enddo

! ---> Variables propres a la phase continue

is1 = is
do icha = 1, ncharb
  is          = is1+icha
  if1m(icha)  = iscapp(is)
  is          = is1+ncharb+icha
  if2m(icha)  = iscapp(is)
enddo

is = is+1
if3m  = iscapp(is)
if ( ihtco2 .eq. 1) then
  is = is+1
  if3mc2 = iscapp(is)
endif
is = is+1
if4p2m = iscapp(is)
if ( ippmod(icp3pl) .eq. 1 ) then
  is = is+1
  if5m  = iscapp(is)
endif

if ( noxyd .ge. 2 ) then
  is = is+1
  if6m  = iscapp(is)
endif
if ( noxyd .eq. 3 ) then
  is = is+1
  if7m  = iscapp(is)
endif

if ( ieqco2 .ge. 1 ) then
  is = is+1
  iyco2 = iscapp(is)
endif

!   - Interface Code_Saturne
!     ======================
!     Construction de l'indirection entre la numerotation du noyau et XML

if (iihmpr.eq.1) then
  call uicpsc (ncharb, nclacp, noxyd, ippmod,       &
               icp3pl, ieqco2, ihtco2,              &
               ihm, inp, ixch, ixck, ixwt, ih2,     &
               if1m, if2m, if3m, if3mc2, if4p2m,    &
               if5m, if6m, if7m, iyco2)
endif

!===============================================================================
! 2. PROPRIETES PHYSIQUES
!    A RENSEIGNER OBLIGATOIREMENT (sinon pb dans varpos)
!    - PROPRES AUX SCALAIRES   : IPHSCA, IVISLS, ISCAVR
!      Rq : pas de variance associee a un scalaire dans notre cas
!    - PROPRES A LA SUSPENSION : ICP
!===============================================================================

do isc = 1, nscapp

  if ( iscavr(iscapp(isc)).le.0 ) then

! ---- Notre physique particuliere est monophasique
    iphsca(iscapp(isc)) = 1

! ---- Viscosite dynamique de reference relative au scalaire
!      ISCAPP(ISC)
    ivisls(iscapp(isc)) = 0

  endif

enddo

! ---- Bien que l'on soit en enthalpie on conserve un CP constant

iphas = iphsca(ihm)
icp(iphas)    = 0

return
end

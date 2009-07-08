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

subroutine atvarp
!================


!===============================================================================
!  FONCTION  :
!  ---------

!    INIT DES POSITIONS DES VARIABLES POUR LE MODULE ATMOSPHERIQUE
! REMPLISSAGE DES PARAMETRES (DEJA DEFINIS) POUR LES SCALAIRES PP

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
include "ppincl.h"
include "atincl.h"
include "ihmpre.h"

!===============================================================================

! VARIABLES LOCALES

integer        isc, iphas

!===============================================================================
!===============================================================================
! 1. DEFINITION DES POINTEURS
!===============================================================================

! 1.1  Dry atmosphere
! =====================

if ( ippmod(iatmos).eq.1 ) then

! ---- Potential temperature
  itempp = iscapp(1)

endif


! 1.2  Humid atmosphere
! =====================

if ( ippmod(iatmos).eq.2 ) then

  ! ---- liquid potential temperature
  itempl = iscapp(1)
  ! ---- total water content
  itotwt = iscapp(2)
  ! ---- total number of droplets
  intdrp = iscapp(3)

endif

!   - Interface Code_Saturne
!     ======================

if (iihmpr.eq.1) then

  call uiatsc (ippmod, iatmos, itempp, itempl, itotwt, intdrp)
  !==========

endif

!===============================================================================
! 2. PROPRIETES PHYSIQUES
!    A RENSEIGNER OBLIGATOIREMENT (sinon pb dans varpos)
!      IPHSCA, IVISLS, ICP
!===============================================================================

do isc = 1, nscapp

  if ( iscavr(iscapp(isc)).le.0 ) then

    ! ---- Notre physique particuliere est monophasique
    iphsca(iscapp(isc)) = 1

    ! ---- Viscosite dynamique moleculaire constante pour les
    !      scalaires ISCAPP(ISC)
    ivisls(iscapp(isc)) = 0

  endif

enddo

! ---- Cp est constant
iphas      = iphsca(iscapp(1))
icp(iphas) = 0

return
end


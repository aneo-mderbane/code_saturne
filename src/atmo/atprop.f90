!-------------------------------------------------------------------------------

!     This file is part of the Code_Saturne Kernel, element of the
!     Code_Saturne CFD tool.

!     Copyright (C) 1998-2009 EDF S.A., France

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

subroutine atprop &
!================

 ( ipropp , ipppst )

!===============================================================================
!  FONCTION  :
!  ---------

!     INIT DES POSITIONS DES VARIABLES D'ETAT
!            POUR LE MODULE ATMOSPHERIQUE
!         (DANS VECTEURS PROPCE, PROPFA, PROPFB)

!-------------------------------------------------------------------------------
! Arguments
!__________________.____._____.________________________________________________.
!    nom           !type!mode !                   role                         !
!__________________!____!_____!________________________________________________!
! ipropp           ! e  ! <-- ! numero de la derniere propriete                !
!                  !    !     !  (les proprietes sont dans propce,             !
!                  !    !     !   propfa ou prpfb)                             !
! ipppst           ! e  ! <-- ! pointeur indiquant le rang de la               !
!                  !    !     !  derniere grandeur definie aux                 !
!                  !    !     !  cellules (rtp,propce...) pour le              !
!                  !    !     !  post traitement                               !
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
include "atincl.h"
include "ppincl.h"
include "ihmpre.h"

!===============================================================================

! Arguments

integer       ipropp, ipppst

! VARIABLES LOCALES

integer       iprop, ipropc

!===============================================================================
! 1. POSITIONNEMENT DES PROPRIETES : PROPCE, PROPFA, PROPFB
!    Atmospheric modules:  dry and humid atmosphere
!===============================================================================


! ---> Temperature (IPPMOD(IATMOS) = 1 or 2)
!-------------------------------------------------------------------------------

! ---> Definition des pointeurs relatifs aux variables d'etat

iprop = ipropp

iprop  = iprop + 1
itempc = iprop

! ---> Liquid water content (IPPMOD(IATMOS) = 2)
!-------------------------------------------------------------------------------

if ( ippmod(iatmos).eq.2) then
  iprop  = iprop + 1
  iliqwt = iprop
endif


! ----  Nb de variables algebriques (ou d'etat)
!         propre a la physique particuliere NSALPP
!         total NSALTO

nsalpp = iprop - ipropp
nsalto = iprop

! ----  On renvoie IPROPP au cas ou d'autres proprietes devraient
!         etre numerotees ensuite

ipropp = iprop


! ---> Positionnement dans le tableau PROPCE
!      et reperage du rang pour le post-traitement

iprop = nproce

iprop          = iprop + 1
ipproc(itempc) = iprop
ipppst         = ipppst + 1
ipppro(iprop)  = ipppst

if (ippmod(iatmos).eq.2) then

  iprop          = iprop + 1
  ipproc(iliqwt) = iprop
  ipppst         = ipppst + 1
  ipppro(iprop)  = ipppst

endif

nproce = iprop


!   - Interface Code_Saturne
!     ======================

if (iihmpr.eq.1) then

  call uiatpr &
  !==========
( nsalpp, nsalto, ippmod, iatmos, &
  ipppro, ipproc, itempc, iliqwt)

endif

return
end subroutine

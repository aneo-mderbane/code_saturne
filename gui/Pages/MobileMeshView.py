# -*- coding: iso-8859-1 -*-
#
#-------------------------------------------------------------------------------
#
#     This file is part of the Code_Saturne User Interface, element of the
#     Code_Saturne CFD tool.
#
#     Copyright (C) 1998-2009 EDF S.A., France
#
#     contact: saturne-support@edf.fr
#
#     The Code_Saturne User Interface is free software; you can redistribute it
#     and/or modify it under the terms of the GNU General Public License
#     as published by the Free Software Foundation; either version 2 of
#     the License, or (at your option) any later version.
#
#     The Code_Saturne User Interface is distributed in the hope that it will be
#     useful, but WITHOUT ANY WARRANTY; without even the implied warranty
#     of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with the Code_Saturne Kernel; if not, write to the
#     Free Software Foundation, Inc.,
#     51 Franklin St, Fifth Floor,
#     Boston, MA  02110-1301  USA
#
#-------------------------------------------------------------------------------

"""
This module defines the values of reference.

This module contains the following classes and function:
- MobileMeshView
"""

#-------------------------------------------------------------------------------
# Library modules import
#-------------------------------------------------------------------------------

import logging

#-------------------------------------------------------------------------------
# Third-party modules
#-------------------------------------------------------------------------------

from PyQt4.QtCore import *
from PyQt4.QtGui  import *

try:
    import mei
    _have_mei = True
except:
    _have_mei = False

#-------------------------------------------------------------------------------
# Application modules import
#-------------------------------------------------------------------------------

from Base.Toolbox   import GuiParam
from MobileMeshForm  import Ui_MobileMeshForm
from Base.QtPage    import setGreenColor, IntValidator,  ComboModel
from MobileMeshModel import MobileMeshModel

if _have_mei:
    from QMeiEditorView import QMeiEditorView

#-------------------------------------------------------------------------------
# log config
#-------------------------------------------------------------------------------

logging.basicConfig()
log = logging.getLogger("MobileMeshView")
log.setLevel(GuiParam.DEBUG)

#-------------------------------------------------------------------------------
# Main class
#-------------------------------------------------------------------------------

class MobileMeshView(QWidget, Ui_MobileMeshForm):
    """
    Class to open Page.
    """
    def __init__(self, parent, case, browser):
        """
        Constructor
        """
        QWidget.__init__(self, parent)

        Ui_MobileMeshForm.__init__(self)
        self.setupUi(self)

        self.case = case
        self.mdl = MobileMeshModel(self.case)
        self.browser = browser

        # Combo model VISCOSITY
        self.modelVISCOSITY = ComboModel(self.comboBoxVISCOSITY,2,1)

        self.modelVISCOSITY.addItem(self.tr("isotropic"), 'isotrop')
        self.modelVISCOSITY.addItem(self.tr("orthotropic"), 'orthotrop')

        # Combo model MEI
        self.modelMEI = ComboModel(self.comboBoxMEI, 2, 1)

        self.modelMEI.addItem(self.tr("user subroutine USVIMA"), 'user_subroutine')
        self.modelMEI.addItem(self.tr("user formula"), 'user_function')

        # Connections
        self.connect(self.groupBoxALE, SIGNAL("clicked(bool)"), self.slotMethod)
        self.connect(self.lineEditNALINF, SIGNAL("textChanged(const QString &)"), self.slotNalinf)
        self.connect(self.comboBoxVISCOSITY, SIGNAL("activated(const QString&)"), self.slotViscosityType)
        self.connect(self.comboBoxMEI, SIGNAL("activated(const QString&)"), self.slotMEI)
        self.connect(self.pushButtonFormula, SIGNAL("clicked(bool)"), self.slotFormula)
 
        # Validators
        validatorNALINF = IntValidator(self.lineEditNALINF, min=0)
        self.lineEditNALINF.setValidator(validatorNALINF)

        if self.mdl.getMethod() == 'on':
            self.groupBoxALE.setChecked(True)
            checked = True
        else:
            self.groupBoxALE.setChecked(False)
            checked = False

        self.slotMethod(checked)
        
        # Enable / disable formula state 
        self.slotMEI(self.comboBoxMEI.currentText())  
        setGreenColor(self.pushButtonFormula, False)


    @pyqtSignature("bool")
    def slotMethod(self, checked):
        """
        Private slot.

        Activates ALE method.

        @type checked: C{True} or C{False}
        @param checked: if C{True}, shows the QGroupBox ALE parameters
        """
        self.groupBoxALE.setFlat(not checked)
        if checked:
            self.frame.show()
            self.mdl.setMethod ("on")
            nalinf = self.mdl.getSubIterations()
            self.lineEditNALINF.setText(QString(str(nalinf)))
            value = self.mdl.getViscosity()
            self.modelVISCOSITY.setItem(str_model=value)
            value = self.mdl.getMEI()
            self.modelMEI.setItem(str_model=value)
        else:
            self.frame.hide()
            self.mdl.setMethod("off")
        self.browser.configureTree(self.case)


    @pyqtSignature("const QString&")
    def slotNalinf(self, text):
        """
        Input viscosity type of mesh : isotrop or orthotrop.
        """        
        nalinf, ok = text.toInt()
        if self.sender().validator().state == QValidator.Acceptable:
            self.mdl.setSubIterations(nalinf)


    @pyqtSignature("const QString&")
    def slotViscosityType(self, text):
        """
        Input viscosity type of mesh : isotrop or orthotrop.
        """
        self.viscosity_type = self.modelVISCOSITY.dicoV2M[str(text)]
        visco = self.viscosity_type
        self.mdl.setViscosity(visco)
        return visco


    @pyqtSignature("const QString&")
    def slotMEI(self, text):
        """
        MEI 
        """
        MEI = self.modelMEI.dicoV2M[str(text)]
        self.MEI = MEI
        self.mdl.setMEI(MEI)
        # enable disable formula button

        isFormulaButtonEnabled = _have_mei and MEI == 'user_function'
        self.pushButtonFormula.setEnabled(isFormulaButtonEnabled)
        setGreenColor(self.pushButtonFormula, isFormulaButtonEnabled)

        return MEI


    @pyqtSignature("const QString&")
    def slotFormula(self, text):
        """
        Run formula editor.
        """
        exp = self.mdl.getFormula()

        if self.mdl.getViscosity() == 'isotrop':
            if not exp:
                exp = "mesh_vi1 ="
            req = [('mesh_vi1', 'mesh viscosity')]
            exa = "mesh_vi1 = 1000;"
        else:
            if not exp:
                exp = "mesh_vi11 ="
            req = [('mesh_vi1', 'mesh viscosity X'),
                   ('mesh_vi2', 'mesh viscosity Y'),
                   ('mesh_vi3', 'mesh viscosity Z')]
            exa = "mesh_vi1 = 1000;\nmesh_vi2 = 1;\nmesh_vi3 = mesh_vi2;"

        symb = [('x', "X cell's gravity center"),
                ('y', "Y cell's gravity center"),
                ('z', "Z cell's gravity center"),
                ('dt', 'time step'),
                ('t', 'current time'),
                ('iter', 'number of iteration')]

        dialog = QMeiEditorView(self,expression = exp,
                                     required   = req,
                                     symbols    = symb,
                                     examples   = exa)
        if dialog.exec_():
            result = dialog.get_result()
            log.debug("slotFormulaMobileMeshView -> %s" % str(result))
            self.mdl.setFormula(result)
            setGreenColor(self.pushButtonFormula, False)


    def tr(self, text):
        """
        Translation
        """
        return text 

#-------------------------------------------------------------------------------
# End
#-------------------------------------------------------------------------------

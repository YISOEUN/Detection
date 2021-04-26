
import warnings

warnings.filterwarnings(action='ignore')
from gui import Ui_Dialog
from PyQt5 import QtCore, QtGui, QtWidgets

# -------------------------------------------------------------------------------------------------------------
import sys
app = QtWidgets.QApplication(sys.argv)
Dialog = QtWidgets.QDialog()
ui = Ui_Dialog()
ui.setupUi(Dialog)

Dialog.show()
sys.exit(app.exec_())
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'graduation3.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

#GUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import uic

from hybridDetection import hybridDetect

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        super().__init__()

        Dialog.setObjectName("Dialog")
        Dialog.resize(1592, 785)
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(10, 70, 631, 51))
        self.lineEdit.setObjectName("lineEdit")
        # input text
        self.lineEdit.returnPressed.connect(self.inputPath)

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 40, 91, 15))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(670, 70, 111, 51))
        self.pushButton.setObjectName("pushButton")
        # 버튼에 기능을 할당하는 코드
        self.pushButton.clicked.connect(self.inputPath)

        self.scrollArea = QtWidgets.QScrollArea(Dialog)
        self.scrollArea.setGeometry(QtCore.QRect(10, 130, 781, 641))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 779, 639))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.label_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_3.setGeometry(QtCore.QRect(400, 20, 131, 21))
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 141, 16))
        self.label_2.setObjectName("label_2")
        self.label_4 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_4.setGeometry(QtCore.QRect(400, 320, 141, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_5.setGeometry(QtCore.QRect(10, 320, 141, 16))
        self.label_5.setObjectName("label_5")

        self.font3 = self.label_3.font()
        self.font3.setBold(True)
        self.font2 = self.label_2.font()
        self.font2.setBold(True)
        self.font4 = self.label_4.font()
        self.font4.setBold(True)
        self.font5 = self.label_5.font()
        self.font5.setBold(True)

        self.label_10 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_10.setGeometry(QtCore.QRect(400, 50, 351, 251))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_11.setGeometry(QtCore.QRect(400, 350, 351, 251))
        self.label_11.setObjectName("label_11")
        self.label_14 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_14.setGeometry(QtCore.QRect(10, 50, 351, 251))
        self.label_14.setObjectName("label_14")

        # self.scrollArea_1 = QScrollArea()


        self.label_15 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_15.setGeometry(QtCore.QRect(10, 350, 351, 251))
        self.label_15.setObjectName("label_15")
        # self.scrollArea_5 = QScrollArea()


        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea_2 = QtWidgets.QScrollArea(Dialog)
        self.scrollArea_2.setGeometry(QtCore.QRect(800, 130, 781, 641))
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 779, 639))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.label_6 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_6.setGeometry(QtCore.QRect(400, 20, 131, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_7.setGeometry(QtCore.QRect(10, 20, 211, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_8.setGeometry(QtCore.QRect(400, 320, 141, 21))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_9.setGeometry(QtCore.QRect(10, 320, 251, 16))
        self.label_9.setObjectName("label_9")

        self.font6 = self.label_6.font()
        self.font6.setBold(True)
        self.font7 = self.label_7.font()
        self.font7.setBold(True)
        self.font8 = self.label_8.font()
        self.font8.setBold(True)
        self.font9 = self.label_9.font()
        self.font9.setBold(True)

        self.label_12 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_12.setGeometry(QtCore.QRect(400, 50, 351, 251))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_13.setGeometry(QtCore.QRect(400, 340, 351, 251))
        self.label_13.setObjectName("label_13")
        self.label_16 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_16.setGeometry(QtCore.QRect(10, 50, 351, 251))
        self.label_16.setObjectName("label_16")
        # self.scrollArea_3 = QScrollArea()


        self.label_17 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_17.setGeometry(QtCore.QRect(10, 350, 351, 251))
        self.label_17.setObjectName("label_17")
        # self.scrollArea_4 = QScrollArea()


        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)


        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Hybrid Detection"))
        self.label.setText(_translate("Dialog", "Input File path"))
        self.pushButton.setText(_translate("Dialog", "Enter"))
        self.label_3.setText(_translate("Dialog", "Misuse attack chart"))
        self.label_2.setText(_translate("Dialog", "Misuse attack data"))
        self.label_4.setText(_translate("Dialog", "Anomoly attack chart"))
        self.label_5.setText(_translate("Dialog", "Anomoly attack data"))

        self.label_10.setText(_translate("Dialog", "(misuse)"))
        self.label_11.setText(_translate("Dialog", "(anomaly)"))
        self.label_14.setText(_translate("Dialog", "(misuse_result)"))
        self.label_15.setText(_translate("Dialog", "(anomaly_result)"))
        self.label_6.setText(_translate("Dialog", "Misuse ROC curve"))
        self.label_7.setText(_translate("Dialog", "Misuse attack models accuarcy"))
        self.label_8.setText(_translate("Dialog", "Anomoly ROC curve"))
        self.label_9.setText(_translate("Dialog", "Anomoly attack models accuarcy"))

        self.label_12.setText(_translate("Dialog", "(misuse_ROC)"))
        self.label_13.setText(_translate("Dialog", "(anomaly_ROC)"))
        self.label_16.setText(_translate("Dialog", "(misuse_model)"))
        self.label_17.setText(_translate("Dialog", "(anomaly_model)"))

    # read input file
    def inputPath(self) :
        #self.lineedit이름.text()
        #check
        # print(self.lineEdit.text())

        hybridDetect(self.lineEdit.text())

        mis_result = QtGui.QPixmap("misuse.png")
        self.label_10.setPixmap(mis_result.scaled(self.label_10.size(), QtCore.Qt.IgnoreAspectRatio))
        # self.scrollAreaWidgetContents.setWindowIconText(self, mis_text)
        ano_result = QtGui.QPixmap("anomaly.png")
        self.label_11.setPixmap(ano_result.scaled(self.label_11.size(), QtCore.Qt.IgnoreAspectRatio))
        # self.scrollAreaWidgetContents_2.setWindowIconText(self, ano_text)

        mRoc_result = QtGui.QPixmap("misuse_ROC.png")
        self.label_12.setPixmap(mRoc_result.scaled(self.label_12.size(), QtCore.Qt.IgnoreAspectRatio))
        aROC_result = QtGui.QPixmap("anomaly_ROC.png")
        self.label_13.setPixmap(aROC_result.scaled(self.label_13.size(), QtCore.Qt.IgnoreAspectRatio))

        result = ''
        f = open('misuse.txt', 'r')
        results = f.readlines()
        if len(results) > 13 :
            for i in range(13):
                result = result + results[i]
            result = result + "... more in 'misuse.txt'\n"
        else :
            for i in results :
                result = result + i

        result = result + "Total : " + str(len(results)) + "\n"
        self.label_14.setText(result)
        f.close()
        # self.scrollArea_1.setWidget(self.label_14)

        result = ''
        f = open('anomaly.txt', 'r')
        results = f.readlines()
        if len(results) > 13:
            for i in range(13):
                result = result + results[i]
            result = result + "... more in 'anomaly.txt'\n"
        else:
            for i in results:
                result = result + i

        result = result + "Total : " + str(len(results)) + "\n"
        self.label_15.setText(result)
        f.close()
        # self.scrollArea_5.setWidget(self.label_15)

        result = ''
        f = open('misuse_model.txt', 'r')
        results = f.readlines()
        for i in results:
            result = result + i
        self.label_16.setText(result)
        f.close()
        # self.scrollArea_3.setWidget(self.label_16)

        result = ''
        f = open('anomaly_model.txt', 'r')
        results = f.readlines()
        for i in results:
            result = result + i
        self.label_17.setText(result)
        f.close()
        # self.scrollArea_4.setWidget(self.label_17)

        #버튼이랑 editline 비활성화

    def showGraph(self,Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        mis_result = QtGui.QPixmap("misuse.png")
        self.label_10.setPixmap(mis_result.scaled(self.label_10.size(), QtCore.Qt.IgnoreAspectRatio))
        ano_result = QtGui.QPixmap("anomaly.png")
        self.label_11.setPixmap(ano_result.scaled(self.label_11.size(), QtCore.Qt.IgnoreAspectRatio))
        mRoc_result = QtGui.QPixmap("misuse_ROC.png")
        self.label_12.setPixmap(mis_result.scaled(self.label_12.size(), QtCore.Qt.IgnoreAspectRatio))
        aROC_result = QtGui.QPixmap("anomaly_ROC.png")
        self.label_13.setPixmap(mis_result.scaled(self.label_13.size(), QtCore.Qt.IgnoreAspectRatio))
        result = ''
        f = open('misuse.txt','r')
        result = "f.readlines()"
        self.label_14.setText(_translate("Dialog",result))
        f.close()

        result = ''
        f = open('anomaly.txt','r')
        result = "f.readlines()"
        self.label_15.setText(_translate("Dialog",result))
        f.close()

        result = ''
        f = open('misuse_model.txt', 'r')
        result = f.readlines()
        self.label_16.setText(_translate("Dialog",result))
        f.close()

        result = ''
        f = open('anomaly_model.txt', 'r')
        result = f.readlines()
        self.label_17.setText(_translate("Dialog",result))
        f.close()





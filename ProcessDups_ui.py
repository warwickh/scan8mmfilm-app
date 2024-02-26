# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/ProcessDups.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(858, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 822, 435))
        self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(778, 435))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.lblImage = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.lblImage.setGeometry(QtCore.QRect(10, 10, 761, 411))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblImage.sizePolicy().hasHeightForWidth())
        self.lblImage.setSizePolicy(sizePolicy)
        self.lblImage.setMinimumSize(QtCore.QSize(600, 0))
        self.lblImage.setObjectName("lblImage")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_2.addWidget(self.scrollArea)
        self.lblInfo = QtWidgets.QLabel(self.centralwidget)
        self.lblInfo.setObjectName("lblInfo")
        self.verticalLayout_2.addWidget(self.lblInfo)
        self.dsbUpper = QtWidgets.QDoubleSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dsbUpper.sizePolicy().hasHeightForWidth())
        self.dsbUpper.setSizePolicy(sizePolicy)
        self.dsbUpper.setDecimals(3)
        self.dsbUpper.setMaximum(1.0)
        self.dsbUpper.setSingleStep(0.001)
        self.dsbUpper.setObjectName("dsbUpper")
        self.verticalLayout_2.addWidget(self.dsbUpper)
        self.dsbLower = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsbLower.setDecimals(3)
        self.dsbLower.setMaximum(1.0)
        self.dsbLower.setSingleStep(0.001)
        self.dsbLower.setObjectName("dsbLower")
        self.verticalLayout_2.addWidget(self.dsbLower)
        self.gbDupButtons = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gbDupButtons.sizePolicy().hasHeightForWidth())
        self.gbDupButtons.setSizePolicy(sizePolicy)
        self.gbDupButtons.setMaximumSize(QtCore.QSize(16777215, 50))
        self.gbDupButtons.setTitle("")
        self.gbDupButtons.setObjectName("gbDupButtons")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.gbDupButtons)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pbtnStart = QtWidgets.QPushButton(self.gbDupButtons)
        self.pbtnStart.setObjectName("pbtnStart")
        self.horizontalLayout_3.addWidget(self.pbtnStart)
        self.pbtnRestart = QtWidgets.QPushButton(self.gbDupButtons)
        self.pbtnRestart.setObjectName("pbtnRestart")
        self.horizontalLayout_3.addWidget(self.pbtnRestart)
        self.pbtnSwap = QtWidgets.QPushButton(self.gbDupButtons)
        self.pbtnSwap.setObjectName("pbtnSwap")
        self.horizontalLayout_3.addWidget(self.pbtnSwap)
        self.pbtnNotDup = QtWidgets.QPushButton(self.gbDupButtons)
        self.pbtnNotDup.setObjectName("pbtnNotDup")
        self.horizontalLayout_3.addWidget(self.pbtnNotDup)
        self.pbtnDup = QtWidgets.QPushButton(self.gbDupButtons)
        self.pbtnDup.setObjectName("pbtnDup")
        self.horizontalLayout_3.addWidget(self.pbtnDup)
        self.pbtnDelete = QtWidgets.QPushButton(self.gbDupButtons)
        self.pbtnDelete.setObjectName("pbtnDelete")
        self.horizontalLayout_3.addWidget(self.pbtnDelete)
        self.pbtnRename = QtWidgets.QPushButton(self.gbDupButtons)
        self.pbtnRename.setObjectName("pbtnRename")
        self.horizontalLayout_3.addWidget(self.pbtnRename)
        self.verticalLayout_2.addWidget(self.gbDupButtons)
        self.gbBrkbuttons = QtWidgets.QGroupBox(self.centralwidget)
        self.gbBrkbuttons.setTitle("")
        self.gbBrkbuttons.setObjectName("gbBrkbuttons")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.gbBrkbuttons)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.rbtnDup = QtWidgets.QRadioButton(self.gbBrkbuttons)
        self.rbtnDup.setObjectName("rbtnDup")
        self.horizontalLayout_2.addWidget(self.rbtnDup)
        self.rbtnScene = QtWidgets.QRadioButton(self.gbBrkbuttons)
        self.rbtnScene.setObjectName("rbtnScene")
        self.horizontalLayout_2.addWidget(self.rbtnScene)
        self.pbtnSkip = QtWidgets.QPushButton(self.gbBrkbuttons)
        self.pbtnSkip.setObjectName("pbtnSkip")
        self.horizontalLayout_2.addWidget(self.pbtnSkip)
        self.pbtnTagError = QtWidgets.QPushButton(self.gbBrkbuttons)
        self.pbtnTagError.setObjectName("pbtnTagError")
        self.horizontalLayout_2.addWidget(self.pbtnTagError)
        self.verticalLayout_2.addWidget(self.gbBrkbuttons)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 858, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lblImage.setText(_translate("MainWindow", "Image"))
        self.lblInfo.setText(_translate("MainWindow", "Info"))
        self.pbtnStart.setText(_translate("MainWindow", "Start"))
        self.pbtnRestart.setText(_translate("MainWindow", "Restart"))
        self.pbtnSwap.setText(_translate("MainWindow", "Swap"))
        self.pbtnNotDup.setText(_translate("MainWindow", "Not Dup"))
        self.pbtnDup.setText(_translate("MainWindow", "Dup"))
        self.pbtnDelete.setText(_translate("MainWindow", "DeleteAll"))
        self.pbtnRename.setText(_translate("MainWindow", "RenameAll"))
        self.rbtnDup.setText(_translate("MainWindow", "Dup"))
        self.rbtnScene.setText(_translate("MainWindow", "Scene"))
        self.pbtnSkip.setText(_translate("MainWindow", "Skip"))
        self.pbtnTagError.setText(_translate("MainWindow", "Error"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))

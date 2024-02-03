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
        MainWindow.resize(800, 600)
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
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 778, 435))
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
        self.dsbLimit = QtWidgets.QDoubleSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dsbLimit.sizePolicy().hasHeightForWidth())
        self.dsbLimit.setSizePolicy(sizePolicy)
        self.dsbLimit.setDecimals(3)
        self.dsbLimit.setMaximum(1.0)
        self.dsbLimit.setSingleStep(0.001)
        self.dsbLimit.setObjectName("dsbLimit")
        self.verticalLayout_2.addWidget(self.dsbLimit)
        self.gbButtons = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gbButtons.sizePolicy().hasHeightForWidth())
        self.gbButtons.setSizePolicy(sizePolicy)
        self.gbButtons.setMaximumSize(QtCore.QSize(16777215, 50))
        self.gbButtons.setTitle("")
        self.gbButtons.setObjectName("gbButtons")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.gbButtons)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pbtnRestart = QtWidgets.QPushButton(self.gbButtons)
        self.pbtnRestart.setObjectName("pbtnRestart")
        self.horizontalLayout_3.addWidget(self.pbtnRestart)
        self.pbtnSwap = QtWidgets.QPushButton(self.gbButtons)
        self.pbtnSwap.setObjectName("pbtnSwap")
        self.horizontalLayout_3.addWidget(self.pbtnSwap)
        self.pbtnNotDup = QtWidgets.QPushButton(self.gbButtons)
        self.pbtnNotDup.setObjectName("pbtnNotDup")
        self.horizontalLayout_3.addWidget(self.pbtnNotDup)
        self.pbtnDup = QtWidgets.QPushButton(self.gbButtons)
        self.pbtnDup.setObjectName("pbtnDup")
        self.horizontalLayout_3.addWidget(self.pbtnDup)
        self.pbtnSkip = QtWidgets.QPushButton(self.gbButtons)
        self.pbtnSkip.setObjectName("pbtnSkip")
        self.horizontalLayout_3.addWidget(self.pbtnSkip)
        self.verticalLayout_2.addWidget(self.gbButtons)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
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
        self.pbtnRestart.setText(_translate("MainWindow", "Restart"))
        self.pbtnSwap.setText(_translate("MainWindow", "Swap"))
        self.pbtnNotDup.setText(_translate("MainWindow", "Not Dup"))
        self.pbtnDup.setText(_translate("MainWindow", "Dup"))
        self.pbtnSkip.setText(_translate("MainWindow", "Skip"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))

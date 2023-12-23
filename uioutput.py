# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/Scan8mmFilm_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1050, 878)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_1.setContentsMargins(-1, 5, -1, -1)
        self.verticalLayout_1.setObjectName("verticalLayout_1")
        self.gboxScanCrop = QtWidgets.QGroupBox(self.centralwidget)
        self.gboxScanCrop.setMinimumSize(QtCore.QSize(250, 0))
        self.gboxScanCrop.setMaximumSize(QtCore.QSize(250, 16777215))
        self.gboxScanCrop.setObjectName("gboxScanCrop")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.gboxScanCrop)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.rbtnScan = QtWidgets.QRadioButton(self.gboxScanCrop)
        self.rbtnScan.setChecked(True)
        self.rbtnScan.setObjectName("rbtnScan")
        self.horizontalLayout_16.addWidget(self.rbtnScan)
        self.rbtnCrop = QtWidgets.QRadioButton(self.gboxScanCrop)
        self.rbtnCrop.setObjectName("rbtnCrop")
        self.horizontalLayout_16.addWidget(self.rbtnCrop)
        self.verticalLayout_1.addWidget(self.gboxScanCrop)
        self.verticalLayout_Buttons = QtWidgets.QVBoxLayout()
        self.verticalLayout_Buttons.setContentsMargins(9, -1, -1, -1)
        self.verticalLayout_Buttons.setObjectName("verticalLayout_Buttons")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.pbtnPrevious = QtWidgets.QPushButton(self.centralwidget)
        self.pbtnPrevious.setMinimumSize(QtCore.QSize(100, 0))
        self.pbtnPrevious.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pbtnPrevious.setAutoRepeat(True)
        self.pbtnPrevious.setObjectName("pbtnPrevious")
        self.horizontalLayout_11.addWidget(self.pbtnPrevious)
        self.pbtnNext = QtWidgets.QPushButton(self.centralwidget)
        self.pbtnNext.setMinimumSize(QtCore.QSize(100, 0))
        self.pbtnNext.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pbtnNext.setAutoRepeat(True)
        self.pbtnNext.setObjectName("pbtnNext")
        self.horizontalLayout_11.addWidget(self.pbtnNext)
        self.verticalLayout_Buttons.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.pbtnStart = QtWidgets.QPushButton(self.centralwidget)
        self.pbtnStart.setEnabled(True)
        self.pbtnStart.setMinimumSize(QtCore.QSize(100, 0))
        self.pbtnStart.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pbtnStart.setObjectName("pbtnStart")
        self.horizontalLayout_12.addWidget(self.pbtnStart)
        self.pbtnStop = QtWidgets.QPushButton(self.centralwidget)
        self.pbtnStop.setEnabled(True)
        self.pbtnStop.setMinimumSize(QtCore.QSize(100, 0))
        self.pbtnStop.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pbtnStop.setObjectName("pbtnStop")
        self.horizontalLayout_12.addWidget(self.pbtnStop)
        self.verticalLayout_Buttons.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.pbtnRandom = QtWidgets.QPushButton(self.centralwidget)
        self.pbtnRandom.setMinimumSize(QtCore.QSize(100, 0))
        self.pbtnRandom.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pbtnRandom.setAutoRepeat(True)
        self.pbtnRandom.setAutoRepeatInterval(1000)
        self.pbtnRandom.setObjectName("pbtnRandom")
        self.horizontalLayout_13.addWidget(self.pbtnRandom)
        self.pbtnMakeFilm = QtWidgets.QPushButton(self.centralwidget)
        self.pbtnMakeFilm.setMinimumSize(QtCore.QSize(100, 0))
        self.pbtnMakeFilm.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pbtnMakeFilm.setObjectName("pbtnMakeFilm")
        self.horizontalLayout_13.addWidget(self.pbtnMakeFilm)
        self.verticalLayout_Buttons.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.pbtnRewind = QtWidgets.QPushButton(self.centralwidget)
        self.pbtnRewind.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pbtnRewind.setObjectName("pbtnRewind")
        self.horizontalLayout_15.addWidget(self.pbtnRewind)
        self.pbtnSpool = QtWidgets.QPushButton(self.centralwidget)
        self.pbtnSpool.setEnabled(True)
        self.pbtnSpool.setMinimumSize(QtCore.QSize(100, 0))
        self.pbtnSpool.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pbtnSpool.setDefault(False)
        self.pbtnSpool.setFlat(False)
        self.pbtnSpool.setObjectName("pbtnSpool")
        self.horizontalLayout_15.addWidget(self.pbtnSpool)
        self.verticalLayout_Buttons.addLayout(self.horizontalLayout_15)
        self.verticalLayout_1.addLayout(self.verticalLayout_Buttons)
        self.gboxAdjust = QtWidgets.QGroupBox(self.centralwidget)
        self.gboxAdjust.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gboxAdjust.sizePolicy().hasHeightForWidth())
        self.gboxAdjust.setSizePolicy(sizePolicy)
        self.gboxAdjust.setMinimumSize(QtCore.QSize(250, 230))
        self.gboxAdjust.setMaximumSize(QtCore.QSize(250, 16777215))
        self.gboxAdjust.setSizeIncrement(QtCore.QSize(0, 0))
        self.gboxAdjust.setBaseSize(QtCore.QSize(0, 0))
        self.gboxAdjust.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.gboxAdjust.setAutoFillBackground(False)
        self.gboxAdjust.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.gboxAdjust.setFlat(True)
        self.gboxAdjust.setCheckable(False)
        self.gboxAdjust.setObjectName("gboxAdjust")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.gboxAdjust)
        self.verticalLayout_4.setContentsMargins(2, 0, 2, -1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setContentsMargins(9, 7, 8, -1)
        self.verticalLayout_3.setSpacing(4)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.comboBox = QtWidgets.QComboBox(self.gboxAdjust)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout_3.addWidget(self.comboBox)
        self.rbtnPosition = QtWidgets.QRadioButton(self.gboxAdjust)
        self.rbtnPosition.setMaximumSize(QtCore.QSize(80, 16777215))
        self.rbtnPosition.setChecked(True)
        self.rbtnPosition.setObjectName("rbtnPosition")
        self.verticalLayout_3.addWidget(self.rbtnPosition)
        self.rbtnSize = QtWidgets.QRadioButton(self.gboxAdjust)
        self.rbtnSize.setMaximumSize(QtCore.QSize(80, 16777215))
        self.rbtnSize.setObjectName("rbtnSize")
        self.verticalLayout_3.addWidget(self.rbtnSize)
        self.pbtnUp = QtWidgets.QPushButton(self.gboxAdjust)
        self.pbtnUp.setAutoRepeat(True)
        self.pbtnUp.setObjectName("pbtnUp")
        self.verticalLayout_3.addWidget(self.pbtnUp)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pbtnLeft = QtWidgets.QPushButton(self.gboxAdjust)
        self.pbtnLeft.setAutoRepeat(True)
        self.pbtnLeft.setObjectName("pbtnLeft")
        self.horizontalLayout.addWidget(self.pbtnLeft)
        self.pbtnRight = QtWidgets.QPushButton(self.gboxAdjust)
        self.pbtnRight.setAutoRepeat(True)
        self.pbtnRight.setObjectName("pbtnRight")
        self.horizontalLayout.addWidget(self.pbtnRight)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.pbtnDown = QtWidgets.QPushButton(self.gboxAdjust)
        self.pbtnDown.setAutoRepeat(True)
        self.pbtnDown.setObjectName("pbtnDown")
        self.verticalLayout_3.addWidget(self.pbtnDown)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.pbtnLedMinus = QtWidgets.QPushButton(self.gboxAdjust)
        self.pbtnLedMinus.setObjectName("pbtnLedMinus")
        self.horizontalLayout_5.addWidget(self.pbtnLedMinus)
        self.pbtnLedPlus = QtWidgets.QPushButton(self.gboxAdjust)
        self.pbtnLedPlus.setObjectName("pbtnLedPlus")
        self.horizontalLayout_5.addWidget(self.pbtnLedPlus)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.verticalLayout_1.addWidget(self.gboxAdjust)
        self.lblHist = QtWidgets.QLabel(self.centralwidget)
        self.lblHist.setMinimumSize(QtCore.QSize(250, 0))
        self.lblHist.setMaximumSize(QtCore.QSize(250, 16777215))
        self.lblHist.setText("")
        self.lblHist.setObjectName("lblHist")
        self.verticalLayout_1.addWidget(self.lblHist)
        spacerItem = QtWidgets.QSpacerItem(117, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_1.addItem(spacerItem)
        self.horizontalLayout_2.addLayout(self.verticalLayout_1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName("gridLayout")
        self.lblScanInfo = QtWidgets.QLabel(self.centralwidget)
        self.lblScanInfo.setText("")
        self.lblScanInfo.setObjectName("lblScanInfo")
        self.gridLayout.addWidget(self.lblScanInfo, 1, 1, 1, 1)
        self.lblScanFrame = QtWidgets.QLabel(self.centralwidget)
        self.lblScanFrame.setMinimumSize(QtCore.QSize(0, 22))
        self.lblScanFrame.setText("")
        self.lblScanFrame.setObjectName("lblScanFrame")
        self.gridLayout.addWidget(self.lblScanFrame, 1, 0, 1, 1)
        self.lblInfo1 = QtWidgets.QLabel(self.centralwidget)
        self.lblInfo1.setText("")
        self.lblInfo1.setObjectName("lblInfo1")
        self.gridLayout.addWidget(self.lblInfo1, 0, 2, 1, 1)
        self.edlFilmName = QtWidgets.QLineEdit(self.centralwidget)
        self.edlFilmName.setFrame(False)
        self.edlFilmName.setObjectName("edlFilmName")
        self.gridLayout.addWidget(self.edlFilmName, 0, 0, 1, 1)
        self.lblInfo2 = QtWidgets.QLabel(self.centralwidget)
        self.lblInfo2.setText("")
        self.lblInfo2.setObjectName("lblInfo2")
        self.gridLayout.addWidget(self.lblInfo2, 1, 2, 1, 1)
        self.lblCropInfo = QtWidgets.QLabel(self.centralwidget)
        self.lblCropInfo.setText("")
        self.lblCropInfo.setObjectName("lblCropInfo")
        self.gridLayout.addWidget(self.lblCropInfo, 0, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 3)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.lblHoleCrop = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblHoleCrop.sizePolicy().hasHeightForWidth())
        self.lblHoleCrop.setSizePolicy(sizePolicy)
        self.lblHoleCrop.setMinimumSize(QtCore.QSize(100, 100))
        self.lblHoleCrop.setText("")
        self.lblHoleCrop.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.lblHoleCrop.setObjectName("lblHoleCrop")
        self.verticalLayout_5.addWidget(self.lblHoleCrop)
        self.gboxAnalyse = QtWidgets.QGroupBox(self.centralwidget)
        self.gboxAnalyse.setMinimumSize(QtCore.QSize(140, 0))
        self.gboxAnalyse.setMaximumSize(QtCore.QSize(140, 16777215))
        self.gboxAnalyse.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.gboxAnalyse.setObjectName("gboxAnalyse")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.gboxAnalyse)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayoutAnalyse = QtWidgets.QVBoxLayout()
        self.verticalLayoutAnalyse.setObjectName("verticalLayoutAnalyse")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.lblOuterThresh = QtWidgets.QLabel(self.gboxAnalyse)
        self.lblOuterThresh.setObjectName("lblOuterThresh")
        self.horizontalLayout_9.addWidget(self.lblOuterThresh)
        self.lblInnerThresh = QtWidgets.QLabel(self.gboxAnalyse)
        self.lblInnerThresh.setObjectName("lblInnerThresh")
        self.horizontalLayout_9.addWidget(self.lblInnerThresh)
        self.verticalLayoutAnalyse.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.dsbOuter = QtWidgets.QDoubleSpinBox(self.gboxAnalyse)
        self.dsbOuter.setMaximum(1.0)
        self.dsbOuter.setSingleStep(0.01)
        self.dsbOuter.setObjectName("dsbOuter")
        self.horizontalLayout_8.addWidget(self.dsbOuter)
        self.dsbInner = QtWidgets.QDoubleSpinBox(self.gboxAnalyse)
        self.dsbInner.setMaximum(1.0)
        self.dsbInner.setSingleStep(0.01)
        self.dsbInner.setObjectName("dsbInner")
        self.horizontalLayout_8.addWidget(self.dsbInner)
        self.verticalLayoutAnalyse.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_xstart = QtWidgets.QHBoxLayout()
        self.horizontalLayout_xstart.setObjectName("horizontalLayout_xstart")
        self.pbtnX1Minus = QtWidgets.QPushButton(self.gboxAnalyse)
        self.pbtnX1Minus.setObjectName("pbtnX1Minus")
        self.horizontalLayout_xstart.addWidget(self.pbtnX1Minus)
        self.pbtnX2Minus = QtWidgets.QPushButton(self.gboxAnalyse)
        self.pbtnX2Minus.setObjectName("pbtnX2Minus")
        self.horizontalLayout_xstart.addWidget(self.pbtnX2Minus)
        self.verticalLayoutAnalyse.addLayout(self.horizontalLayout_xstart)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.pbtnX1Plus = QtWidgets.QPushButton(self.gboxAnalyse)
        self.pbtnX1Plus.setObjectName("pbtnX1Plus")
        self.horizontalLayout_7.addWidget(self.pbtnX1Plus)
        self.pbtnX2Plus = QtWidgets.QPushButton(self.gboxAnalyse)
        self.pbtnX2Plus.setObjectName("pbtnX2Plus")
        self.horizontalLayout_7.addWidget(self.pbtnX2Plus)
        self.verticalLayoutAnalyse.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.lblX1 = QtWidgets.QLabel(self.gboxAnalyse)
        self.lblX1.setObjectName("lblX1")
        self.horizontalLayout_10.addWidget(self.lblX1)
        self.lblX2 = QtWidgets.QLabel(self.gboxAnalyse)
        self.lblX2.setObjectName("lblX2")
        self.horizontalLayout_10.addWidget(self.lblX2)
        self.verticalLayoutAnalyse.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_6.addLayout(self.verticalLayoutAnalyse)
        self.verticalLayout_5.addWidget(self.gboxAnalyse)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem1)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_3.addWidget(self.line)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 611, 757))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollAreaWidgetContents.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents.setSizePolicy(sizePolicy)
        self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(400, 300))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.lblImage = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblImage.sizePolicy().hasHeightForWidth())
        self.lblImage.setSizePolicy(sizePolicy)
        self.lblImage.setMinimumSize(QtCore.QSize(200, 100))
        self.lblImage.setFrameShape(QtWidgets.QFrame.Panel)
        self.lblImage.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lblImage.setText("")
        self.lblImage.setTextFormat(QtCore.Qt.PlainText)
        self.lblImage.setScaledContents(False)
        self.lblImage.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.lblImage.setObjectName("lblImage")
        self.horizontalLayout_4.addWidget(self.lblImage)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout_3.addWidget(self.scrollArea)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 10)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1050, 19))
        self.menubar.setObjectName("menubar")
        self.menuMain = QtWidgets.QMenu(self.menubar)
        self.menuMain.setObjectName("menuMain")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionSelect_Film_Folder = QtWidgets.QAction(MainWindow)
        self.actionSelect_Film_Folder.setObjectName("actionSelect_Film_Folder")
        self.actionSelect_Scan_Folder = QtWidgets.QAction(MainWindow)
        self.actionSelect_Scan_Folder.setObjectName("actionSelect_Scan_Folder")
        self.actionSelect_Crop_Folder = QtWidgets.QAction(MainWindow)
        self.actionSelect_Crop_Folder.setObjectName("actionSelect_Crop_Folder")
        self.actionClear_Scan_Folder = QtWidgets.QAction(MainWindow)
        self.actionClear_Scan_Folder.setObjectName("actionClear_Scan_Folder")
        self.actionClear_Crop_Folder = QtWidgets.QAction(MainWindow)
        self.actionClear_Crop_Folder.setObjectName("actionClear_Crop_Folder")
        self.menuMain.addAction(self.actionAbout)
        self.menuMain.addAction(self.actionExit)
        self.menuFile.addAction(self.actionSelect_Film_Folder)
        self.menuFile.addAction(self.actionSelect_Scan_Folder)
        self.menuFile.addAction(self.actionSelect_Crop_Folder)
        self.menuFile.addAction(self.actionClear_Scan_Folder)
        self.menuFile.addAction(self.actionClear_Crop_Folder)
        self.menubar.addAction(self.menuMain.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "8mm Film Scanner"))
        self.gboxScanCrop.setTitle(_translate("MainWindow", "Operation mode"))
        self.rbtnScan.setText(_translate("MainWindow", "Scan"))
        self.rbtnCrop.setText(_translate("MainWindow", "Crop"))
        self.pbtnPrevious.setText(_translate("MainWindow", "Previous"))
        self.pbtnNext.setText(_translate("MainWindow", "Next"))
        self.pbtnStart.setText(_translate("MainWindow", "Start"))
        self.pbtnStop.setText(_translate("MainWindow", "Stop"))
        self.pbtnRandom.setText(_translate("MainWindow", "Random"))
        self.pbtnMakeFilm.setText(_translate("MainWindow", "Make Film"))
        self.pbtnRewind.setText(_translate("MainWindow", "Rewind"))
        self.pbtnSpool.setText(_translate("MainWindow", "Spool"))
        self.gboxAdjust.setTitle(_translate("MainWindow", "Adjust"))
        self.rbtnPosition.setText(_translate("MainWindow", "Position"))
        self.rbtnSize.setText(_translate("MainWindow", "Size"))
        self.pbtnUp.setText(_translate("MainWindow", "Up"))
        self.pbtnLeft.setText(_translate("MainWindow", "Left"))
        self.pbtnRight.setText(_translate("MainWindow", "Right"))
        self.pbtnDown.setText(_translate("MainWindow", "Down"))
        self.pbtnLedMinus.setText(_translate("MainWindow", "LED-"))
        self.pbtnLedPlus.setText(_translate("MainWindow", "LED+"))
        self.edlFilmName.setPlaceholderText(_translate("MainWindow", "<Enter film name>"))
        self.gboxAnalyse.setTitle(_translate("MainWindow", "Analyse"))
        self.lblOuterThresh.setText(_translate("MainWindow", "Outer"))
        self.lblInnerThresh.setText(_translate("MainWindow", "Inner"))
        self.pbtnX1Minus.setText(_translate("MainWindow", "x1<"))
        self.pbtnX2Minus.setText(_translate("MainWindow", "x2<"))
        self.pbtnX1Plus.setText(_translate("MainWindow", "x1>"))
        self.pbtnX2Plus.setText(_translate("MainWindow", "x2>"))
        self.lblX1.setText(_translate("MainWindow", "x1"))
        self.lblX2.setText(_translate("MainWindow", "x2"))
        self.menuMain.setTitle(_translate("MainWindow", "Main"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionSelect_Film_Folder.setText(_translate("MainWindow", "Select Film Folder"))
        self.actionSelect_Scan_Folder.setText(_translate("MainWindow", "Select Scan Folder"))
        self.actionSelect_Crop_Folder.setText(_translate("MainWindow", "Select Crop Folder"))
        self.actionClear_Scan_Folder.setText(_translate("MainWindow", "Clear Scan Folder"))
        self.actionClear_Crop_Folder.setText(_translate("MainWindow", "Clear Crop Folder"))

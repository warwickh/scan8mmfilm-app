from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5 import QtCore, QtWidgets
#from FilmScanModule import Ini, Camera, Frame, Film, getAdjustableRects, getAnalysisTypes
import sys
from time import sleep
import cv2
import string
from ProcessDups_ui import Ui_MainWindow
import os

class Window(QMainWindow, Ui_MainWindow):
    
    sigToCropTread = pyqtSignal(int)
    sigToScanTread = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)


    def connectSignalsSlots(self):
        self.pbtnStart.clicked.connect(self.start)
        self.pbtnStop.clicked.connect(self.stop)
        self.pbtnUp.clicked.connect(self.up)
        self.pbtnDown.clicked.connect(self.down)
        self.pbtnLeft.clicked.connect(self.left)
        self.pbtnRight.clicked.connect(self.right)
        self.rbtnScan.toggled.connect(self.modeChanged)
        self.pbtnNext.clicked.connect(self.nnext)
        self.pbtnPrevious.clicked.connect(self.previous)
        self.pbtnRandom.clicked.connect(self.random)
        self.pbtnMakeFilm.clicked.connect(self.makeFilm)
        self.pbtnLedPlus.clicked.connect(self.ledPlus)
        self.pbtnLedMinus.clicked.connect(self.ledMinus)
        self.pbtnSpool.clicked.connect(self.spool)
        self.dsbOuter.valueChanged.connect(self.outerThreshChanged)
        self.dsbInner.valueChanged.connect(self.innerThreshChanged)
        self.pbtnX1Minus.clicked.connect(self.x1Minus)
        self.pbtnX1Plus.clicked.connect(self.x1Plus)
        self.pbtnX2Minus.clicked.connect(self.x2Minus)
        self.pbtnX2Plus.clicked.connect(self.x2Plus)
        self.actionExit.triggered.connect(self.doClose)
        self.actionAbout.triggered.connect(self.about)
        self.actionSelect_Film_Folder.triggered.connect(self.selectFilmFolder)
        self.actionSelect_Scan_Folder.triggered.connect(self.selectScanFolder)
        self.actionSelect_Crop_Folder.triggered.connect(self.selectCropFolder)
        self.actionClear_Scan_Folder.triggered.connect( self.clearScanFolder)
        self.actionClear_Crop_Folder.triggered.connect( self.clearCropFolder)

if __name__ == "__main__":
    app = QApplication(sys.argv) 
    win = Window()
    win.show()
    sys.exit(app.exec())
    sys.exit(0)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5 import QtCore, QtGui
from PyQt5 import QtCore, QtWidgets
#from FilmScanModule import Ini, Camera, Frame, Film, getAdjustableRects, getAnalysisTypes
import sys
from time import sleep
import cv2
import string
from ProcessDups_ui import Ui_MainWindow
from DupFrameDetector import DupFrameDetector
import os
import csv

class Window(QMainWindow, Ui_MainWindow):
    
    sigToCropTread = pyqtSignal(int)
    sigToScanTread = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
<<<<<<< HEAD
        self.currentLimit = 0.94
=======
        self.currentLimit = 0.99
>>>>>>> 5f3dffff76af9c4d7e7b6fb73631a1d85022b8ef
        self.similarity=0
        self.currentImg = 1
        self.results = []
        self.img1pth = ""
        self.img2pth = ""
        self.detector = DupFrameDetector()
<<<<<<< HEAD
        self.cropFolder = os.path.expanduser("~/scanframes/crop/roll4")
=======
        self.cropFolder = os.path.expanduser("~/scanframes/crop/roll3")
>>>>>>> 5f3dffff76af9c4d7e7b6fb73631a1d85022b8ef
        self.resultPath = os.path.join(self.cropFolder,"dup_results.csv")
        self.connectSignalsSlots()
        self.doLblImagePrep = False
        self.swapTimer = QTimer()
        self.swapTimer.timeout.connect(self.swap)
        QTimer.singleShot(100, self.initData)

    def initData(self):
        if not self.selectCropFolder(text="Please select a folder for cropped frames", init=False):
            self.doClose()
            return
        print(f"Starting at {self.cropFolder}")
        self.resultPath = os.path.join(self.cropFolder,"dup_results.csv")
        self.detector.createDupLog(self.cropFolder)
        if self.detector.loaded:
            self.getNextImages()
            self.loadImage(self.img1)
            self.updateInfoPanel()
            self.dsbLimit.setValue(self.currentLimit)
            self.swapTimer.start(500)

    def selectCropFolder(self, *, text="Select Crop Folder", init=True):
        dir = QtWidgets.QFileDialog.getExistingDirectory(caption=text, directory=os.path.commonpath([self.cropFolder]))
        if dir:
            self.cropFolder = os.path.abspath(dir)
            if init:
                self.updateInfoPanel()
            return True
        return False

    def connectSignalsSlots(self):
        self.pbtnRestart.clicked.connect(self.restart)
        self.pbtnSwap.clicked.connect(self.swap)
        self.pbtnDup.clicked.connect(self.dup)
        self.pbtnNotDup.clicked.connect(self.notDup)
        self.pbtnSkip.clicked.connect(self.skip)
<<<<<<< HEAD
        self.pbtnDelete.clicked.connect(self.delete)
=======
>>>>>>> 5f3dffff76af9c4d7e7b6fb73631a1d85022b8ef
        self.dsbLimit.valueChanged.connect(self.limitChanged)

    def updateInfoPanel(self):
        self.lblInfo.setText(f"Current frame {self.currentImg} {self.img2pth} {self.similarity:03f} logged {len(self.results)}")
        
    def limitChanged(self):
        self.currentLimit = self.dsbLimit.value()
        self.updateInfoPanel()

    def restart(self):
        print("Restarting file")
        self.initData()
        open(self.resultPath, 'w').close()
    
    def getCVImage(self, imagePath):
        if not os.path.exists(imagePath):
            print(f"No file {imagePath}")
        else:
            return cv2.imread(imagePath)        

    def getNextImages(self):
        while True:
            self.currentRow = self.detector.getNextRow()
            self.img1pth = self.currentRow['img1']
            self.img2pth = self.currentRow['img2']
            self.img1 = self.getCVImage(self.img1pth)
            self.img2 = self.getCVImage(self.img2pth)
            self.similarity = float(self.currentRow['similarity'])
            self.updateInfoPanel()
            print(f"{self.img1pth} {self.img2pth} {self.similarity}")
            if self.similarity>=self.currentLimit:
                break

    def skip(self):
<<<<<<< HEAD
        self.pbtnDup.setEnabled(False)
        self.pbtnNotDup.setEnabled(False)
        self.getNextImages()
        self.loadImage(self.img1)
        self.updateInfoPanel()
        self.pbtnDup.setEnabled(True)
        self.pbtnNotDup.setEnabled(True)
        

    def delete(self):
        self.pbtnDelete.setEnabled(False)
        self.detector.deleteAll(self.cropFolder)
        self.pbtnDelete.setEnabled(True)
=======
        self.getNextImages()
        self.loadImage(self.img1)
        self.updateInfoPanel()
>>>>>>> 5f3dffff76af9c4d7e7b6fb73631a1d85022b8ef

    def swap(self):
        if self.currentImg==2:
            self.loadImage(self.img1)
            self.currentImg=1
        else:
            self.loadImage(self.img2)
            self.currentImg=2
        self.updateInfoPanel()
        #print(f"swap to {self.currentImg}")

    def dup(self):
        result = {'img1':self.img1pth, 'img2':self.img2pth, 'similarity': self.similarity, 'isDup': True}
<<<<<<< HEAD
        self.pbtnDup.setEnabled(False)
        self.pbtnNotDup.setEnabled(False)
=======
>>>>>>> 5f3dffff76af9c4d7e7b6fb73631a1d85022b8ef
        self.results.append(result)
        self.saveRow(result)
        self.getNextImages()
        self.loadImage(self.img1)
        self.updateInfoPanel()
<<<<<<< HEAD
        self.pbtnDup.setEnabled(True)
        self.pbtnNotDup.setEnabled(True)

    def notDup(self):
        result = {'img1':self.img1pth, 'img2':self.img2pth, 'similarity': self.similarity, 'isDup': False}
        self.pbtnDup.setEnabled(False)
        self.pbtnNotDup.setEnabled(False)
=======

    def notDup(self):
        result = {'img1':self.img1pth, 'img2':self.img2pth, 'similarity': self.similarity, 'isDup': False}
>>>>>>> 5f3dffff76af9c4d7e7b6fb73631a1d85022b8ef
        self.results.append(result)
        self.saveRow(result)
        self.getNextImages()
        self.loadImage(self.img1)
        self.updateInfoPanel()
<<<<<<< HEAD
        self.pbtnDup.setEnabled(True)
        self.pbtnNotDup.setEnabled(True)
        
=======

>>>>>>> 5f3dffff76af9c4d7e7b6fb73631a1d85022b8ef
    def loadImage(self, img):
        self.prepLblImage()
        self.lblImage.setPixmap(self.getQPixmap(img,self.scrollAreaWidgetContents))
        self.updateInfoPanel()

    def saveRow(self, rowDict):
        with open(self.resultPath, 'a') as csvfile:
            fieldNames=['img1','img2','similarity','isDup']
            writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
            #writer.writeheader()
            writer.writerow(rowDict)

    def convert_cv_qt(self, cv_img, dest=None):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        if dest is None:
            return QtGui.QPixmap.fromImage(convert_to_Qt_format) 
        else:
            im = convert_to_Qt_format.scaled(dest.width(), dest.height(), QtCore.Qt.KeepAspectRatio)
            return QtGui.QPixmap.fromImage(im)
        
    def getQPixmap(self, img, dest=None):
        return self.convert_cv_qt(img, dest)
        
    def getCropped(self, dest=None):
        ScaleFactor = int(self.ScaleFactor)
        outCrop = cv2.resize(self.imageCropped, (0,0), fx=1/ScaleFactor, fy=1/ScaleFactor)
        return self.convert_cv_qt(outCrop, dest)

    def showImage(self):
        self.prepLblImage()
        self.lblImage.setPixmap(self.frame.getCropOutline(self.scrollAreaWidgetContents) )
        self.lblHoleCrop.setPixmap(self.frame.getHoleCrop())
        #if self.frame.histogram is not None:
        self.lblHist.setPixmap(self.frame.getHistogram())

    def prepLblImage(self):
        if self.doLblImagePrep and self.horizontalLayout_4.SizeConstraint != QtWidgets.QLayout.SetMinimumSize :
            self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
            self.lblImage.clear()
            self.doLblImagePrep = False

if __name__ == "__main__":
    app = QApplication(sys.argv) 
    win = Window()
    win.show()
    sys.exit(app.exec())
    sys.exit(0)
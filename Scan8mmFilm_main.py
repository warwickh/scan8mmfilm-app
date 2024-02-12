# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5 import QtCore, QtWidgets
from FilmScanModule import Ini, Camera, Frame, Film, getAdjustableRects, getAnalysisTypes
import sys
from time import sleep
import cv2
import string
from Scan8mmFilm_ui import Ui_MainWindow
import os

try:
    from picamera2 import Picamera2
    from picamera2.previews.qt import QGlPicamera2
    from libcamera import Transform
    import piDeviceInterface as pidevi
    picamera2_present = True
except ImportError:                                                                                                                                                                                            
    picamera2_present = False
 

if picamera2_present:
    os.environ["LIBCAMERA_LOG_LEVELS"] = "2"
    tuning = Picamera2.load_tuning_file("imx219_noir.json")
    picam2 = Picamera2(tuning=tuning)
    #picam2 = Picamera2()
    #preview_config = picam2.create_preview_configuration(main={"size": (Camera.ViewWidth, Camera.ViewHeight)},
    #    transform=Transform(vflip=True,hflip=True))
    preview_config = picam2.create_preview_configuration(main={"size": (3280, 2464)},
        transform=Transform(vflip=True,hflip=True))
    picam2.configure(preview_config)
    
    pidevi.initGpio()

class Window(QMainWindow, Ui_MainWindow):
    
    sigToCropTread = pyqtSignal(int)
    sigToScanTread = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        Ini.loadConfig()
        if picamera2_present:
            self.lblImage.hide() 
            # self.qpicamera2 = QGlPicamera2(picam2, width=3280, height=2464, keep_ar=True)
            self.qpicamera2 = QGlPicamera2(picam2, width=800, height=600, keep_ar=True)
            self.horizontalLayout_4.addWidget(self.qpicamera2)
        self.dsbInner.setValue(float(Frame.innerThresh))
        self.dsbOuter.setValue(float(Frame.outerThresh))
        self.sbWT.setValue(int(Frame.whiteThreshold))
        self.sbHsvMargin.setValue(Frame.hsvMargin)
        #print(f"Init str(Frame.analysisType) {str(Frame.analysisType)}")
        self.connectSignalsSlots()
        self.lblHoleCrop.setMinimumWidth(Frame.getHoleCropWidth())
        self.adjustableRects = getAdjustableRects()
        for r in self.adjustableRects:
            self.comboBox.addItem(r.name)
        self.analysisTypes = getAnalysisTypes()
        for r in self.analysisTypes:
            self.cbAnalysisType.addItem(r)
        self.adjRectIx = 0
        self.comboBox.currentIndexChanged.connect(self.adjustableRectChanged)
        self.cbAnalysisType.setCurrentText(str(Frame.analysisType))
        self.cbAnalysisType.currentTextChanged.connect(self.analysisTypeChanged)

        
        #self.dsbInner.setValue(float(Frame.innerThresh))
        #self.dsbOuter.setValue(float(Frame.outerThresh))
        self.doLblImagePrep = False

        QTimer.singleShot(100, self.initScanner)

    def connectSignalsSlots(self):
        self.pbtnStart.clicked.connect(self.start)
        self.pbtnStop.clicked.connect(self.stop)
        self.pbtnUp.clicked.connect(self.up)
        self.pbtnDown.clicked.connect(self.down)
        self.pbtnLeft.clicked.connect(self.left)
        self.pbtnRight.clicked.connect(self.right)
        self.rbtnScan.toggled.connect(self.modeChanged)
        self.rbtnS8.toggled.connect(self.formatChanged)
        
        self.pbtnNext.clicked.connect(self.nnext)
        self.pbtnPrevious.clicked.connect(self.previous)
        self.pbtnRandom.clicked.connect(self.random)
        self.pbtnMakeFilm.clicked.connect(self.makeFilm)
        #self.pbtnLedPlus.clicked.connect(self.ledPlus)
        #self.pbtnLedMinus.clicked.connect(self.ledMinus)
        self.pbtnApplyWT.clicked.connect(self.whiteThresholdApply)
        self.pbtnDetectWT.clicked.connect(self.whiteThresholdDetect)
        self.pbtnSpool.clicked.connect(self.spool)
        self.dsbOuter.valueChanged.connect(self.outerThreshChanged)
        self.dsbInner.valueChanged.connect(self.innerThreshChanged)
        #self.sbWT.valueChanged.connect(self.whiteThresholdChanged)
        self.sbHsvMargin.valueChanged.connect(self.hsvMarginChanged)
        self.pbtnX1Minus.clicked.connect(self.x1Minus)
        self.pbtnX1Plus.clicked.connect(self.x1Plus)
        self.pbtnX2Minus.clicked.connect(self.x2Minus)
        self.pbtnX2Plus.clicked.connect(self.x2Plus)
        self.actionExit.triggered.connect(self.doClose)
        self.actionAbout.triggered.connect(self.about)
        if  picamera2_present:
            print("picam why?")
            self.qpicamera2.done_signal.connect(self.capture_done)

        self.actionSelect_Film_Folder.triggered.connect(self.selectFilmFolder)
        self.actionSelect_Scan_Folder.triggered.connect(self.selectScanFolder)
        self.actionSelect_Crop_Folder.triggered.connect(self.selectCropFolder)
        self.actionClear_Scan_Folder.triggered.connect( self.clearScanFolder)
        self.actionClear_Crop_Folder.triggered.connect( self.clearCropFolder)

    # Menu actions --------------------------------------------------------------------------------------------

    def selectFilmFolder(self, *, text="Select Film Folder"):
        dir = QtWidgets.QFileDialog.getExistingDirectory(caption=text, directory=os.path.commonpath([Film.filmFolder]))
        if dir:
            Film.filmFolder = os.path.abspath(dir)
            return True
        return False
    

    def selectScanFolder(self, *, text="Select Scan Folder", init=True):
        dir = QtWidgets.QFileDialog.getExistingDirectory(caption=text, directory=os.path.commonpath([Film.scanFolder]))
        if dir:
            Film.scanFolder = os.path.abspath(dir)
            if init:
                QTimer.singleShot(100, self.initScanner)
            return True
        return False

    def selectCropFolder(self, *, text="Select Crop Folder", init=True):
        dir = QtWidgets.QFileDialog.getExistingDirectory(caption=text, directory=os.path.commonpath([Film.cropFolder]))
        if dir:
            Film.cropFolder = os.path.abspath(dir)
            if init:
                self.updateInfoPanel()
            return True
        return False

    def clearScanFolder(self):
        button = QMessageBox.question(self, "Delete",  f"Delete all {Film.getScanCount()} .jpg files in {Film.scanFolder}?",QMessageBox.Yes|QMessageBox.No)
        if button == QMessageBox.Yes:
            Film.deleteFilesInFolder(Film.scanFolder)
            self.initScanner()
       
    def clearCropFolder(self):
        button = QMessageBox.question(self, "Delete", f"Delete all {Film.getCropCount()} .jpg files in {Film.cropFolder}?",QMessageBox.Yes|QMessageBox.No)
        if button == QMessageBox.Yes:
            Film.deleteFilesInFolder(Film.cropFolder)
            self.updateInfoPanel()

    def about(self):
        QMessageBox.about(
            self,
            "8mm Film Scanner App",
            "<p>Built with:</p>"
            "<p>- PyQt5</p>"
            "<p>- Qt Designer</p>"
            "<p>- Python 3.9</p>"
            "<p>- OpenCV</p>",
        )

    def doClose(self):
        self.close()

    # Button actions ---------------------------------------------------------------------------------------------------------------
    
    def formatChanged(self):
        if self.rbtnS8.isChecked():
            print("Selected S8")
            Frame.format = "s8"
            self.film.format = "s8"
        else:
            print("Selected R8")
            Frame.format = "r8"
            self.film.format = "r8"
        self.adjustableRects = getAdjustableRects()
        self.comboBox.clear()
        for r in self.adjustableRects:
            self.comboBox.addItem(r.name)
        self.modeChanged()

    def modeChanged(self):
        self.prepLblImage()
        if self.rbtnScan.isChecked():
            self.showInfo("Mode Scan")
            if picamera2_present:            
                print("picam why2")
                self.lblImage.hide()
                self.qpicamera2.show()
            if self.frame is None:
                self.frame = self.film.getFirstFrame()
            self.showFrame()  
        else:
            self.showInfo("Mode Crop")
            self.lblImage.show()
            if picamera2_present:            
                self.qpicamera2.hide() 
                print("picam why3")
            if self.frame is not None and self.frame.imagePathName is not None:
                self.refreshFrame()
            else:
                self.frame = self.film.getFirstFrame()
                self.showFrame()  

    def start(self):
        self.showInfo("start")
        self.prepLblImage()
        self.film.name = self.edlFilmName.text()
        self.film.curFrameNo = -1
        self.frame = self.film.getNextFrame()
        if self.rbtnScan.isChecked():
            if self.frame is not None:
                self.showScan()
            if picamera2_present:   
                self.motorStart()
                self.startScanFilm()
        else:
            self.startCropAll()
           
    def stop(self):
        self.showInfo("Stop")
        try:
            if self.rbtnScan.isChecked():
                if picamera2_present:
                    self.sigToScanTread.emit(0)
                    self.threadScan.running = False
                    sleep(0.2)
            else:
                self.sigToCropTread.emit(0)
                self.threadCrop.running = False
                sleep(0.2)
            self.enableButtons(busy=False)
        except:
            pass
    
    def nnext(self):
        self.showInfo("Next")
        if self.rbtnScan.isChecked():
            if picamera2_present: 
                self.enableButtons(busy=True)  
                self.motorStart()
                pidevi.spoolFwd(0.02)
                pidevi.stepCw(self.film.stepsPrFrame)
                #pidevi.spoolStart()
                self.showHoleCrop()
        else:
            self.frame = self.film.getNextFrame()
            self.showFrame()

    def previous(self):
        self.showInfo("Previous")
        if self.rbtnScan.isChecked():
            if picamera2_present: 
                self.enableButtons(busy=True)  
                self.motorStart()
                pidevi.spoolBack(0.02)
                pidevi.stepCcw(self.film.stepsPrFrame)
                #pidevi.spoolStart()
                self.showHoleCrop()
        else:
            self.frame = self.film.getPreviousFrame()
            self.showFrame()
            
    def random(self):
        self.showInfo("Random")
        self.frame = self.film.getRandomFrame()
        self.showFrame()
                                         
    def makeFilm(self):
        self.showInfo("Make Film")  
        if not os.path.exists(Film.filmFolder):
            if not self.selectFilmFolder("Please Select a Film Folder"):
                 return
        fileName = self.toValidFileName(self.edlFilmName.text())
        if len(fileName) == 0 :
            self.showInfo("Enter a valid filneme!")  
        else:
            self.enableButtons(busy=True)
            self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
            self.doLblImagePrep = True
            self.lblImage.clear()
            self.lblScanFrame.setText("")
            self.film.name = fileName
            self.messageText = ""
            self.film.makeFilm(self.film.name, self.filmMessage, self.filmDone)
        
    def down(self):
        if self.rbtnScan.isChecked():
            if picamera2_present:
                self.enableButtons(busy=True)  
                self.motorStart()
                pidevi.spoolBack(0.05)
                pidevi.stepCcw(4)
                #pidevi.spoolStart()
                self.showHoleCrop()
        else:
            if self.rbtnPosition.isChecked() :
                self.adjustableRects[self.adjRectIx].adjY(10)
            else:
                self.adjustableRects[self.adjRectIx].adjYSize(10)
            self.refreshFrame() 
            self.showAdjustValues()
            
    def up(self):
        if self.rbtnScan.isChecked():
            if picamera2_present:
                self.enableButtons(busy=True)  
                self.motorStart()
                pidevi.spoolFwd(0.05)
                pidevi.stepCw(4)  
                #pidevi.spoolStart()
                self.showHoleCrop()
        else:
            if self.rbtnPosition.isChecked() :
                self.adjustableRects[self.adjRectIx].adjY(-10)
            else:
                self.adjustableRects[self.adjRectIx].adjYSize(-10)
            self.refreshFrame()
            self.showAdjustValues()
            
    def left(self):
        if self.rbtnCrop.isChecked():
            if self.rbtnPosition.isChecked() :
                self.adjustableRects[self.adjRectIx].adjX(-10)
            else:
                self.adjustableRects[self.adjRectIx].adjXSize(-10)
            self.refreshFrame()
            self.showAdjustValues()
            
    def right(self):
        if self.rbtnCrop.isChecked():
            if self.rbtnPosition.isChecked() :
                self.adjustableRects[self.adjRectIx].adjX(10)
            else:
                self.adjustableRects[self.adjRectIx].adjXSize(10)
            self.refreshFrame()
            self.showAdjustValues()

    def adjustableRectChanged(self, i):
        self.adjRectIx = i
        self.showAdjustValues()

    def analysisTypeChanged(self, i):
        self.analysisType = i
        Frame.analysisType = i
        print(f"Setting analysis type to {Frame.analysisType}")
        self.refreshFrame()
        self.showAdjustValues()

    def whiteThresholdDetect(self):
        Frame.whiteThreshold = self.frame.getWhiteThreshold()
        self.sbWT.setValue(int(Frame.whiteThreshold))
        print(f"Setting white threshold to {Frame.whiteThreshold} from image")

    def whiteThresholdApply(self):
        Frame.whiteThreshold = self.sbWT.value()
        self.refreshFrame()
        self.showAdjustValues()

    def hsvMarginChanged(self):
        Frame.hsvMargin = self.sbHsvMargin.value()
        self.refreshFrame()
        self.showAdjustValues()

    def outerThreshChanged(self):
        Frame.outerThresh = self.dsbOuter.value()
        self.refreshFrame()
        self.showAdjustValues()
        
    def innerThreshChanged(self):
        Frame.innerThresh = self.dsbInner.value()
        self.refreshFrame()
        self.showAdjustValues()
        
    #def ledPlus(self):
    #    if self.rbtnScan.isChecked():
    #        if picamera2_present:
    #            Film.led_dc = pidevi.ledPlus()
    #            print(f"LED DC now {Film.led_dc}")

    #def ledMinus(self):
    #    if self.rbtnScan.isChecked():
    #        if picamera2_present:
    #            Film.led_dc = pidevi.ledMinus()
    #            print(f"LED DC now {Film.led_dc}")

    def x1Plus(self):
        Frame.ratioX1+=20
        self.lblX1.setText(str(Frame.ratioX1))
        self.refreshFrame()
        self.showAdjustValues()

    def x1Minus(self):
        Frame.ratioX1-=20
        self.lblX1.setText(str(Frame.ratioX1))
        self.refreshFrame()
        self.showAdjustValues()
                
    def x2Plus(self):
        Frame.ratioX2+=20
        self.lblX2.setText(str(Frame.ratioX2))
        self.refreshFrame()
        self.showAdjustValues()
                
    def x2Minus(self):
        Frame.ratioX2-=20
        self.lblX2.setText(str(Frame.ratioX2))
        self.refreshFrame()
        self.showAdjustValues()
    
    def spool(self):
        if picamera2_present:
            pidevi.spool()
            
    # Process or timer actions ---------------------------------------------------------------------------------------------------------

    def motorTimeout(self) :
        self.motorTicks = self.motorTicks + 1 
        #pidevi.spoolStart()
        if self.motorTicks > 10 :
            self.motorStop()
        
    def capture_done(self,job):
        image = picam2.wait(job)
        print("picture taken!")
        sleep(0.5)
        #image = cv2.resize(image, (640, 480))
        self.frame = Frame(image=image)
        self.frame.calcCrop()
        print("in capture_done")
        self.lblHoleCrop.setPixmap(self.frame.getHoleCrop())
        #self.lblHist.setPixmap(cv2.resize(self.frame.getHistogram(), (0,0), fx=0.5, fy=0.5))
        #self.lblHist.setPixmap(self.frame.getHistogram())
        self.updateInfoPanel()
        self.motorTicks = 0
        print("only the buttons to go")   
        if self.scanDone :
            self.enableButtons(busy=False)
        print("capture_done finished")
            
    def scanProgress(self, info, i, frame ):
        self.lblScanInfo.setText(info)
        self.motorTicks = 0   
        if frame is not None:
            if self.lblImage.isVisible():
                self.lblImage.setPixmap(frame.getCropped())
            self.lblHoleCrop.setPixmap(frame.getHoleCrop())
            self.lblHist.setPixmap(frame.getHistogram())
            #print(f"Resizing hist to {self.lblHist.size}")
            #self.lblHist.setPixmap(cv2.resize(frame.getHistogram(), (0,0), fx=0.5, fy=0.5))
            self.frame = frame

    def cropProgress(self, info, i, frame):
        self.lblCropInfo.setText(info)
        if frame is not None:
            if self.lblImage.isVisible():
                self.lblImage.setPixmap(frame.getCropped())
            self.lblHoleCrop.setPixmap(frame.getHoleCrop())
            self.lblHist.setPixmap(frame.getHistogram())
            self.frame = frame

    def cropStateChange(self, info, res):
        self.updateInfoPanel()        
        self.showInfo(info)
        self.enableButtons(busy=False)
    
    def resultToText(self, res):
        if res == 0:
            return "Hole found"
        elif res == 1: 
            return "Hole not found"
        elif res == 2: 
            return "Hole to large. No film?" # hole to large
        elif res == 3: 
            return "Hole malformed - no center"
        else:
            return "" # e.g. -1


    def scanStateChange(self, info, result):
        self.updateInfoPanel()
        self.showInfo(info + self.resultToText(result))
        self.scanDone = True
        self.enableButtons(busy=False)
        self.motorStop()
            
    def filmMessage(self, s):
        self.messageText = self.messageText + "\n" + s
        self.lblImage.setText(self.messageText) 
            
    def filmDone(self):
        self.enableButtons(busy=False)
    
    # Shared GUI control methods ------------------------------------------------------------------------------------------------------------------------------

    def enableButtons(self, *, busy=False):
        idle = not busy
        pi = picamera2_present
        scan = self.rbtnScan.isChecked()
        crop = self.rbtnCrop.isChecked()
        frame = self.frame is not None # True if scan folder has frames

        self. rbtnR8.setEnabled(True)
        self. rbtnS8.setEnabled(True)

        self.pbtnStart.setEnabled(idle and (scan or frame))
        self.pbtnStop.setEnabled(busy)
        self.pbtnMakeFilm.setEnabled(idle and crop and frame)

        self.pbtnNext.setEnabled(idle and (pi or (crop and frame)))
        self.pbtnPrevious.setEnabled(idle and (pi or (crop and frame)))
        self.pbtnRandom.setEnabled(idle and crop and frame)
        
        self.pbtnUp.setEnabled(idle and (pi or (crop and frame)))
        self.pbtnDown.setEnabled(idle and (pi or (crop and frame)))
        self.pbtnLeft.setEnabled(idle and crop and frame)
        self.pbtnRight.setEnabled(idle and crop and frame)

        self.rbtnScan.setEnabled(idle and pi)
        self.rbtnCrop.setEnabled(idle)
        
        self.pbtnMakeFilm.setEnabled(idle and crop and frame)
        #self.chkRewind.setEnabled(pi and crop)
        self.menuFile.setEnabled(idle)
        self.edlFilmName.setEnabled(idle)

        self.comboBox.setEnabled(frame)#idle and crop and frame)
        self.rbtnPosition.setEnabled(idle and crop and frame)
        self.rbtnSize.setEnabled(idle and crop and frame)
        self.pbtnX1Minus.setEnabled(frame)
        self.pbtnX1Plus.setEnabled(frame)
        self.pbtnX2Minus.setEnabled(frame)
        self.pbtnX2Plus.setEnabled(frame)
        self.dsbOuter.setEnabled(frame)
        self.dsbInner.setEnabled(frame)
        self.sbWT.setEnabled(frame)
        #self.pbtnLedPlus.setEnabled(scan)
        #self.pbtnLedMinus.setEnabled(scan)
        self.pbtnApplyWT.setEnabled(frame)
        self.pbtnDetectWT.setEnabled(frame)
        #self.pbtnSetWT.setEnabled(idle and crop and frame)
        self.pbtnSpool.setEnabled(pi and (crop or scan))

    # Shared GUI update methods ---------------------------------------------------------------------------------------------------------------------------     

    def initScanner(self):
        if not os.path.exists(Film.scanFolder):
            if not self.selectScanFolder(text="Please Select a Scan folder", init=False):
                 self.doClose()
                 return

        if not os.path.exists(Film.cropFolder):
            if not self.selectCropFolder(text="Please select a folder for cropped frames", init=False):
                self.doClose()
                return
                    
        self.film = Film("")
        self.frame = None
        self.lblImage.clear()
        self.lblHoleCrop.clear()
        if Frame.format == "s8":
            self.rbtnS8.setChecked(True)
        else:
            self.rbtnR8.setChecked(True)
        if picamera2_present:
            self.timer = QTimer()
            self.timer.timeout.connect(self.motorTimeout)
            self.scanDone = True
            # Start in Scan mode
            if self.rbtnScan.isChecked():
                self.modeChanged() # was set - force action
            else:
                self.rbtnScan.setChecked(True)
        else:
            # Start in Crop Mode
            if self.rbtnCrop.isChecked():
                self.modeChanged() # was set - force action
            else:
                self.rbtnCrop.setChecked(True) 
        self.enableButtons(busy=False)
        #self.lblX1.setText(str(Frame.ratioX1))
        #self.lblX2.setText(str(Frame.ratioX2))

    def updateInfoPanel(self):
        self.lblCropInfo.setText(f"Cropped frame count {Film.getCropCount()}")
        #self.edlMinHoleArea.setText("N/A")#str(Frame.holeMinArea))
        if self.frame is not None:
            frame = self.frame
            self.lblScanInfo.setText(f"Frame {self.film.curFrameNo} of {self.film.scanFileCount}")
            if self.frame.imagePathName is None:
                self.lblScanFrame.setText(Film.scanFolder)
            else:
                self.lblScanFrame.setText(self.frame.imagePathName)       
            self.lblInfo1.setText(f"cX={frame.cX} cY={frame.cY} midy={frame.midy}")
            if frame.sprocketSize is not None:
                self.lblInfo2.setText(f"res={frame.locateHoleResult} sprocketSize={frame.sprocketSize}")
        else:
            self.lblScanInfo.setText(f"Frame count = {self.film.scanFileCount}")
            self.lblScanFrame.setText(Film.scanFolder) 
            self.lblInfo1.setText("")
            self.lblInfo2.setText("")

    def showAdjustValues(self):
        rect = self.adjustableRects[self.adjRectIx]
        ar = "" if self.adjRectIx != 0 else f"aspect ratio={rect.getXSize()/rect.getYSize():.2f}" 
        self.statusbar.showMessage(f"Adjusted: {rect.name} x1={rect.x1} y1={rect.y1}  x2={rect.x2} y2={rect.y2} width={rect.x2-rect.x1} height={rect.y2-rect.y1} {ar}")
    
    def refreshFrame(self):
        if self.frame is not None and self.frame.imagePathName is not None:
            self.frame = Frame(self.frame.imagePathName)
        self.showFrame()
  
    def showFrame(self):
        if self.rbtnScan.isChecked():
            if self.frame is not None:
                self.showScan()
        else:
            if self.frame is not None:
                self.showCrop() 
        self.updateInfoPanel()
        self.enableButtons()
        
    def showScan(self):
        if picamera2_present:  
            self.showHoleCrop()
        else:
            self.prepLblImage()
            self.lblImage.setPixmap(self.frame.getQPixmap(self.scrollAreaWidgetContents) )
        self.lblHoleCrop.update()
        if self.frame.histogram is not None:
            self.lblHist.setPixmap(self.frame.getHistogram())
        
    def showCrop(self):
        self.prepLblImage()
        self.lblImage.setPixmap(self.frame.getCropOutline(self.scrollAreaWidgetContents) )
        self.lblHoleCrop.setPixmap(self.frame.getHoleCrop())
        #if self.frame.histogram is not None:
        self.lblHist.setPixmap(self.frame.getHistogram())

    def showInfo(self,text):
        self.statusbar.showMessage(text)

    def prepLblImage(self):
        if self.doLblImagePrep and self.horizontalLayout_4.SizeConstraint != QtWidgets.QLayout.SetMinimumSize :
            self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
            self.lblImage.clear()
            self.doLblImagePrep = False
 
    def showHoleCrop(self): 
        if picamera2_present:
            self.enableButtons(busy=True)
            print(f"Setting capture config RGB888 size: {Camera.ViewWidth}, {Camera.ViewHeight}")
            capture_config = picam2.create_still_configuration(main={"format": "RGB888","size": (Camera.ViewWidth, Camera.ViewHeight)},transform=Transform(vflip=True,hflip=True))
            # This causes a call to capture_done when done 
            picam2.switch_mode_and_capture_array(capture_config, "main", signal_function=self.qpicamera2.signal_done)

    # Shared device control ---------------------------------------------------------------------------------------------------------------------------------------
    
    def motorStart(self):
        if self.rbtnScan.isChecked():
            self.motorTicks = 0
            self.timer.start(2000)
            pidevi.startScanner()
            #pidevi.spoolStart()

    def motorStop(self):
        pidevi.stepStop()
        self.timer.stop()
         
    def startCropAll(self):
        self.enableButtons(busy=True)
        self.lblScanFrame.setText("")
        self.lblInfo1.setText("")
        self.lblInfo2.setText("")
        self.threadCrop = QThreadCrop(self.film)
        self.sigToCropTread.connect(self.threadCrop.on_source)
        self.threadCrop.sigProgress.connect(self.cropProgress)
        self.threadCrop.sigStateChange.connect(self.cropStateChange)
        self.threadCrop.start()

    def startScanFilm(self):
        self.scanDone = False
        self.enableButtons(busy=True)
        self.lblScanFrame.setText("")
        self.lblInfo1.setText("")
        self.lblInfo2.setText("")
        self.motorStart()
        self.threadScan = QThreadScan(self.film)
        self.sigToScanTread.connect(self.threadScan.on_source)
        
        self.threadScan.sigProgress.connect(self.scanProgress)
        self.threadScan.sigStateChange.connect(self.scanStateChange)
        self.threadScan.start()

    # Shared utility methods -----------------------------------------------------------------------------------------------
         
    def toValidFileName(self, s):
        valid_chars = "-_() %s%s" % (string.ascii_letters, string.digits)
        filename = ''.join(c for c in s if c in valid_chars)
        filename = filename.replace(' ','_') # I don't like spaces in filenames.
        return filename    

# Thread =============================================================================

class QThreadCrop(QtCore.QThread):
    sigProgress = pyqtSignal(str, int, Frame)
    sigStateChange = pyqtSignal(str, int)
    
    def __init__(self, film, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.film = film
        self.cmd = 1 # run
        self.result = "Done"
        
    def on_source(self, cmd):
        if cmd  == 0:
            self.result = "Stopped manually"
        self.cmd = cmd
        
    def progress(self, frame) :
        cnt = self.film.curFrameNo
        self.sigProgress.emit(f"{cnt} frames cropped", cnt, frame)    
        return self.cmd
            
    def run(self):
        self.running = True

        try:
            self.film.cropAll(self.progress)
            sleep(1)    
        except Exception as err:
            self.result = "Exception: " + str(err)
            print("QThreadCrop", err)
            
        self.sigStateChange.emit(self.result, self.cmd)
        self.running = False

# Thread =============================================================================

class QThreadScan(QtCore.QThread):
    sigProgress = pyqtSignal(str, int, Frame)
    sigStateChange = pyqtSignal(str, int)
    
    def __init__(self, film, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.film = film
        self.cmd = 1 # run 
        self.midy = None#self.frame.midy #TODO switch to per frame setting
        self.tolerance = 6
        self.pixelsPerStep = 2.5
        self.parent = parent
        self.frameNo = Film.getFileCount(Film.scanFolder)
        
    def on_source(self, cmd):
        self.cmd = cmd    
        
    def saveFrame(self):
        imgname = os.path.join(Film.scanFolder, 'scan' + format(self.frameNo, '06') + '.jpg')             
        print("pre cap")
        request = picam2.capture_request()
        print("post cap")
        request.save("main", imgname)
        print("imgname", imgname)
        #print(request.get_metadata()) # this is the metadata for this image
        request.release()
        # signal progress
        self.sigProgress.emit(f"{self.frameNo} frames scanned", self.frameNo, self.frame)    
       
    def run(self):
        self.running = True
        pidevi.startScanner()
        self.locateHoleResult = 0
        oldY = 0
        stuckCount = 0
        release = True
        pidevi.spool()
        firstAdj = True
        while self.cmd == 1 :
            try:
                #pidevi.spoolStart()
                print(f"Setting capture config RGB888 size: {Camera.ViewWidth}, {Camera.ViewHeight}")
                capture_config = picam2.create_still_configuration(main={"format": "RGB888","size": (Camera.ViewWidth, Camera.ViewHeight)},transform=Transform(vflip=True,hflip=True))
                image = picam2.switch_mode_and_capture_array(capture_config, "main") #, signal_function=self.qpicamera2.signal_done)
                self.frame = Frame(image=image)
                
                
                locateHoleResult = self.frame.locateSprocketHole()#Frame.holeMinArea)
                print("cY",self.frame.cY ,"oldY", oldY, "locateHoleResult", locateHoleResult,"cmd",self.cmd,"sprocketsize",self.frame.sprocketSize)
                
                if locateHoleResult != 0 :
                    self.cmd = 2
                    self.locateHoleResult = locateHoleResult
                    break
                    
                if oldY != 0 and oldY == self.frame.cY :
                    # adjustment failed - film stuck
                    stuckCount += 1
                    if stuckCount > 200:
                        # film really really stuck
                        self.cmd = 3
                        break
                self.midy = self.frame.midy
                currentcY = self.frame.cY#//self.frame.ScaleFactor
                tolerance = self.tolerance*self.frame.ScaleFactor
                tolstep = int(abs(currentcY-self.frame.midy)//self.frame.ScaleFactor//self.pixelsPerStep)
                tolstep = min(tolstep, 20)
                #tolstep = 4
                print(f"{currentcY}-----------------------------------------------")
                if currentcY > self.midy + tolerance:
                    #self.motorStart()
                    pidevi.spoolFwd(0.05)
                    release = True
                    self.sigProgress.emit(f"{self.frameNo} adjusting up", self.frameNo, self.frame)
                    print(f"{currentcY} lower than {self.midy} so Moving up {abs(currentcY-self.midy)} pixels {tolstep} steps")  
                    pidevi.stepCw(tolstep)
                    sleep(.2)  
                    oldY = currentcY
                    firstAdj = False

                elif currentcY < self.midy - tolerance:
                    if firstAdj and currentcY < self.midy - 3*tolerance:
                        pidevi.stepCw(40)
                    else:
                        self.sigProgress.emit(f"{self.frameNo} adjusting down", self.frameNo, self.frame)  
                        print(f"{currentcY} higher than {self.midy} so Moving down {abs(currentcY-self.midy)} pixels {tolstep} steps")
                        #self.motorStart()
                        if release:
                            pidevi.spoolBack(0.05)
                            release = False
                        #pidevi.adjDn()  
                        pidevi.stepCcw(tolstep)
                        sleep(.2) 
                        oldY = currentcY 
                    firstAdj = False
        
                    
                elif (currentcY <= self.midy + tolerance) and (currentcY >= self.midy - tolerance):
                    self.saveFrame() 
                    print(f"cY {currentcY} midy {self.midy} tol {tolerance} =========================================================")  
                    #self.motorStart()
                    pidevi.spoolFwd(0.15)
                    release = True
                    pidevi.stepCw(self.film.stepsPrFrame)
                    self.frameNo += 1
                    adjustedY = 0
                    stuckCount = 0
                    firstAdj = True

                sleep(0.1)  
                  
            except Exception as err:
                print("QThreadScan", err)
                self.sigStateChange.emit("Exception:" + str(err), -1)
                self.cmd = -1
            
        if self.frameNo == 0:
            # Save at least one frame for adjusting image cropping in crop mode
            self.saveFrame() 
        self.running = False
        if self.cmd == 0:
            self.sigStateChange.emit("Stopped manually", -1)
        elif self.cmd == 2:
            self.sigStateChange.emit("Stopped: ", self.locateHoleResult)
        elif self.cmd == 3:
            self.sigStateChange.emit("Stopped: Film is stuck", -1)

# =============================================================================

if __name__ == "__main__":
    safe = False
    if safe:
        try:
            app = QApplication(sys.argv) 
            win = Window()
            if  picamera2_present:
                picam2.start()
            win.show()
            sys.exit(app.exec())
        except:
            if  picamera2_present:
                pidevi.cleanup()
    else:
        app = QApplication(sys.argv) 
        win = Window()
        win.show()
        sys.exit(app.exec())
        if  picamera2_present:
                pidevi.cleanup()
    Ini.saveConfig()
    sys.exit(0)

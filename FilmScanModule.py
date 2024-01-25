# -*- coding: utf-8 -*-

import os, fnmatch
import cv2
import random
import glob
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QProcess
import configparser
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool as ProcessPool
from SprocketHole import sprocketHole

dbg = 0

inifile = os.path.join(os.path.dirname(__file__),'scanner.ini')
defaultBaseDir = "c:\\data\\film8" if os.sep == "\\" else "/home/pi/film8"

class Ini:
    camera = 'camera'
    paths = 'paths'
    film = 'film'
    frame = 'frame'

    def loadConfig():
        config = configparser.ConfigParser()

        if len(config.read(inifile)) == 1:

            Film.filmFolder = config[Ini.paths]['film_folder']
            Film.scanFolder = config[Ini.paths]['scan_folder']
            Film.cropFolder = config[Ini.paths]['crop_folder']
            Camera.ViewWidth = config[Ini.camera].getint('view_width')
            Camera.ViewHeight = config[Ini.camera].getint('view_height')

            Film.format = config[Ini.film]['format']
            Film.led_dc = config[Ini.film].getint('led_dc')
            Film.resolution = config[Ini.film]['resolution']
            Film.s8_framerate = config[Ini.film].getint('s8_framerate')
            Film.s8_stepsPrFrame = config[Ini.film].getint('s8_steps_pr_frame')
            Film.r8_framerate = config[Ini.film].getint('r8_framerate')
            Film.r8_stepsPrFrame = config[Ini.film].getint('r8_steps_pr_frame')
            
            Frame.format = Film.format
            #Frame.outerThresh = config[Ini.frame].getfloat('outerThresh')
            Frame.innerThresh = config[Ini.frame].getfloat('innerThresh')
            Frame.s8_stdSprocketHeight = config[Ini.frame].getfloat('s8_stdSprocketHeight')
            Frame.s8_ratio = config[Ini.frame].getfloat('s8_ratio')
            Frame.s8_midx = config[Ini.frame].getint('s8_midx')
            Frame.s8_midy = config[Ini.frame].getint('s8_midy')
            Frame.r8_stdSprocketHeight = config[Ini.frame].getfloat('r8_stdSprocketHeight')
            Frame.r8_ratio = config[Ini.frame].getfloat('r8_ratio')
            Frame.r8_midx = config[Ini.frame].getint('r8_midx')
            Frame.r8_midy = config[Ini.frame].getint('r8_midy')
            Frame.s8_frameCrop.load(config)
            Frame.s8_holeCrop.load(config)
            Frame.r8_frameCrop.load(config)
            Frame.r8_holeCrop.load(config)
        else:
            Ini.saveConfig()

        Frame.initScaleFactor()
        
    def saveConfig():
        config = configparser.ConfigParser()

        if not config.has_section(Ini.paths):
            config[Ini.paths] = {}
        config[Ini.paths]['film_folder'] = Film.filmFolder
        config[Ini.paths]['scan_folder'] = Film.scanFolder
        config[Ini.paths]['crop_folder'] = Film.cropFolder
        
        if not config.has_section(Ini.camera):
            config[Ini.camera] = {}
        config[Ini.camera]['view_width'] = str(Camera.ViewWidth)
        config[Ini.camera]['view_height'] = str(Camera.ViewHeight)

        if not config.has_section(Ini.film):
            config[Ini.film] = {}
        config[Ini.film]['format'] = Film.format
        config[Ini.film]['led_dc'] = str(Film.led_dc)
        config[Ini.film]['resolution'] = Film.resolution
        config[Ini.film]['s8_framerate']  = str(Film.s8_framerate)
        config[Ini.film]['s8_steps_pr_frame'] = str(Film.s8_stepsPrFrame)
        config[Ini.film]['r8_framerate'] = str(Film.r8_framerate)
        config[Ini.film]['r8_steps_pr_frame'] = str(Film.r8_stepsPrFrame)

        if not config.has_section(Ini.frame):
            config[Ini.frame] = {}
        config[Ini.frame]['format'] = Film.format
        #config[Ini.frame]['outerThresh'] = str(Frame.outerThresh)
        config[Ini.frame]['innerThresh'] = str(Frame.innerThresh)
        config[Ini.frame]['s8_stdSprocketHeight'] = str(Frame.s8_stdSprocketHeight)
        config[Ini.frame]['s8_ratio'] = str(Frame.s8_ratio)
        config[Ini.frame]['s8_midx'] = str(Frame.s8_midx)
        config[Ini.frame]['s8_midy'] = str(Frame.s8_midy)
        config[Ini.frame]['r8_stdSprocketHeight'] = str(Frame.r8_stdSprocketHeight)
        config[Ini.frame]['r8_ratio'] = str(Frame.r8_ratio)
        config[Ini.frame]['r8_midx'] = str(Frame.r8_midx)
        config[Ini.frame]['r8_midy'] = str(Frame.r8_midy)

        Frame.s8_frameCrop.save(config)
        Frame.s8_holeCrop.save(config)
        Frame.r8_frameCrop.save(config)
        Frame.r8_holeCrop.save(config)

        with open(inifile, 'w') as configfile:
            config.write(configfile)
        
def getAdjustableRects():
    if Film.format == 's8':
        return [Frame.s8_frameCrop, Frame.s8_holeCrop]
    else:
        return [Frame.r8_frameCrop, Frame.r8_holeCrop]
    
class Camera:
    ViewWidth = 3280#1640
    ViewHeight = 2464#1232

class Rect:
    def __init__(self, name, x1, y1, x2, y2):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    def load(self, config):
        self.x1 = config[self.name].getint('x1')
        self.x2 = config[self.name].getint('x2')
        self.y1 = config[self.name].getint('y1')
        self.y2 = config[self.name].getint('y2')

    def save(self, config):
        if not config.has_section(self.name):
            config[self.name] = {}
        config[self.name]['x1'] = str(self.x1)
        config[self.name]['x2'] = str(self.x2)
        config[self.name]['y1'] = str(self.y1) 
        config[self.name]['y2'] = str(self.y2)

    def getXSize(self):
        return self.x2 - self.x1
    
    def getYSize(self):
        return self.y2 - self.y1
    
    def adjX(self, adj):
        #if self.x1 + adj >= 0 :
        self.x1 = int(self.x1 + adj)
        self.x2 = int(self.x2 + adj)

    def adjY(self, adj):
        #if self.y1 + adj >= 0 :
        self.y1 = int(self.y1 + adj)
        self.y2 = int(self.y2 + adj)
        
    def adjXSize(self, adj):
        if self.x2 + adj > self.x1 :
            self.x2 = int(self.x2 + adj)
    
    def adjYSize(self, adj):
        if self.y2 + adj > self.y1 :
            self.y2 = int(self.y2 + adj) 

class Frame:

    format = "s8"
    s8_frameCrop = Rect("s8_frame_crop", 2, -107, 2+1453, 1040-107)
    s8_holeCrop = Rect("s8_hole_crop", 75, 0, 110, 2463) 
    s8_stdSprocketHeight = 0.1
    s8_sprocketWidth = 0.05
    s8_midx = 64
    s8_midy = 240
    r8_frameCrop = Rect("r8_frame_crop", 15, 16, 15+1323, 16+947)
    r8_holeCrop = Rect("r8_hole_crop", 90, 0, 240, 276)
    r8_stdSprocketHeight = 0.1
    r8_sprocketWidth = 0.13
    r8_midx = 64
    r8_midy = 120
    ScaleFactor = 1.0 # overwritten by initScaleFactor()
    outerThresh = 0.52
    innerThresh = 0.2
    s8_ratio = (4.23-1.143)/1.143 #gap/sprocket width
    r8_ratio = (4.23-1.27)/1.143 #gap/sprocket width
    hist_path = os.path.expanduser("~/my_cv2hist_lim.png")
    
    def initScaleFactor():
        Frame.ScaleFactor = Camera.ViewWidth/640.0 

    def getHoleCropWidth():
        if Frame.format == "s8":
            return 1*(Frame.s8_holeCrop.x2-Frame.s8_holeCrop.x1) #wider to capture vertical line also
        else:
            return 1*(Frame.r8_holeCrop.x2-Frame.r8_holeCrop.x1) #wider to capture vertical line also
        #return self.holeCrop.x2 - Frame.holeCrop.x1
    
    def __init__(self, imagePathName=None,*,image=None):
        self.imagePathName = imagePathName
        if image is None and imagePathName is not None :
            self.image = cv2.imread(imagePathName)
        elif image is not None :
            self.image = image
        self.thresh = None
        self.dy,self.dx,_ = self.image.shape
        self.ScaleFactor = self.dx/640.0
        print(f"Init Frame path {imagePathName} Scalefactor {Frame.ScaleFactor}")
        if Frame.format == "s8":
            self.stdSprocketHeight = Frame.s8_stdSprocketHeight
            self.holeCrop = Rect("hole_crop", int(Frame.s8_holeCrop.x1*self.ScaleFactor), 0, int(Frame.s8_holeCrop.x2*self.ScaleFactor), self.dy-1)
            self.frameCrop = Frame.s8_frameCrop
            self.ratio = Frame.s8_ratio
            self.sprocketWidth = Frame.s8_sprocketWidth
            self.midy = int(Frame.s8_midy*self.ScaleFactor)
        
        else:
            self.stdSprocketHeight = Frame.r8_stdSprocketHeight
            self.holeCrop = Rect("hole_crop", int(Frame.r8_holeCrop.x1*self.ScaleFactor), 0, int(Frame.r8_holeCrop.x2*self.ScaleFactor), self.dy-1)
            self.frameCrop = Frame.r8_frameCrop
            self.ratio = Frame.r8_ratio
            self.sprocketWidth = Frame.r8_sprocketWidth
            self.midy = int(Frame.r8_midy*self.ScaleFactor)
        self.midx = 115*self.ScaleFactor   # always overwitten 
        self.cX = self.midx 
        self.cY = self.midy
        self.sprocketHeight = 0    
        self.histogram = None
        #self.histogram = cv2.imread(os.path.expanduser("~/my_cv2hist_lim.png"))
        self.resultImagePath = os.path.expanduser("~/sprocket_hole_result.png")
        self.locateHoleResult = 6
        #print(f"init complete {self.__dict__}")
        #self.whiteThreshold = 225 # always overwritten by call to getWhiteThreshold
        self.sprocketHole = sprocketHole(self)
        
        
    def convert_cv_qt(self, cv_img, dest=None):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        print(f"{rgb_image.shape}")
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        if dest is None:
            return QtGui.QPixmap.fromImage(convert_to_Qt_format) 
        else:
            im = convert_to_Qt_format.scaled(dest.width(), dest.height(), QtCore.Qt.KeepAspectRatio)
            return QtGui.QPixmap.fromImage(im)
        
    def getQPixmap(self, dest=None):
        return self.convert_cv_qt(self.image, dest)
        
    def getCropped(self, dest=None):
        return self.convert_cv_qt(self.imageCropped, dest)
        
    def getHoleCrop(self) :
        #cv2.imwrite(os.path.expanduser("~/getHoleCrop.png"), self.imageHoleCrop)
        self.imageHoleCrop = cv2.resize(cv2.imread(self.resultImagePath), (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)
        #self.imageHoleCrop = cv2.resize(cv2.imread(self.resultImagePath), (0,0), fx=1/10, fy=1/10)
        return self.convert_cv_qt(self.imageHoleCrop)

    def getHistogram(self):
        width = 200
        try:
            self.histogram = cv2.imread(Frame.hist_path)
            y,x,_ = self.histogram.shape
            scale = 200/y
            #self.histogram = cv2.resize(self.histogram, (200, 200))
            return self.convert_cv_qt(cv2.resize(self.histogram, (0,0), fx=scale, fy=scale))
        except Exception as exc:
            print(f"Couldn't read histogram")
            return

    def calcCrop(self):
        #self.locateHoleResult = self.locateSprocketHole(Frame.holeMinArea)
        self.locateHoleResult = self.locateSprocketHole()
        print(f"Calc crop self.cY {self.cY} + Frame.holeCrop.y1 {self.holeCrop.y1} = self.cY + Frame.holeCrop.y1 {self.cY + self.holeCrop.y1}")
        print(f"Frame.ScaleFactor {self.ScaleFactor}")
        print(f"Frame.frameCrop.y1 {self.frameCrop.y1}")
        print(f"int((self.cY + Frame.holeCrop.y1) * Frame.ScaleFactor)+Frame.frameCrop.y1 {int((self.cY + self.holeCrop.y1) * self.ScaleFactor)+self.frameCrop.y1}")
        x = int(self.cX + (self.frameCrop.x1 * Frame.ScaleFactor))
        y = int(self.cY + (self.frameCrop.y1 * Frame.ScaleFactor)) 
        self.p1 = (x, y)
        self.p2 = (x+self.frameCrop.getXSize(), y+self.frameCrop.getYSize())
        print(f"crop self.p1 {self.p1} self.p2 {self.p2}")
        
    def getCropOutline(self, dest=None):
        self.calcCrop()
        cv2.rectangle(self.image, self.p1, self.p2, (0, 255, 0), 10)
        return self.convert_cv_qt(self.image, dest)
        
    def cropPic(self):
        self.calcCrop()
        self.imageCropped = self.image[self.p1[1]:self.p2[1], self.p1[0]:self.p2[0]]   

    def locateSprocketHole(self):
        self.locateHoleResult, self.cY, self.cX = self.sprocketHole.process()#cX rX
        return self.locateHoleResult
           
class Film:
    format = "s8"
    resolution = "720x540"
    s8_framerate = 24
    r8_framerate = 12
    led_dc = 100
    s8_stepsPrFrame = 100 # value for Standard 8
    r8_stepsPrFrame = 80 # value for Standard 8
    filmFolder = os.path.join(defaultBaseDir, "film")
    scanFolder = os.path.join(defaultBaseDir, "scan")
    cropFolder = os.path.join(defaultBaseDir, "crop")

    def __init__(self, name = ""):
        self.name = name
        self.scanFileCount = Film.getFileCount(Film.scanFolder)  # - number of *.jpg files in scan directory
        self.curFrameNo = -1
        self.p = None
        if Film.format == "s8":
            self.stepsPrFrame = Film.s8_stepsPrFrame
            self.framerate = Film.s8_framerate
        else:
            self.stepsPrFrame = Film.r8_stepsPrFrame
            self.framerate = Film.r8_framerate
     
    def getCurrentFrame(self):
        self.curFrameNo -= 1
        return self.getNextFrame()
    
    def getFileList(dir):
        list = [f for f in os.listdir(dir) if 
                    fnmatch.fnmatch(f, "*.jpg") and
                    os.path.isfile(os.path.join(dir, f))]
        if len(list) > 1 :
            return sorted(list)
        return list

    def getFileCount(dir):
        n = 0
        for f in os.listdir(dir):
             if fnmatch.fnmatch(f, "*.jpg") and os.path.isfile(os.path.join(dir, f)):
                n = n + 1
        return n

    def deleteFilesInFolder(dir):
        for f in os.listdir(dir):
             if fnmatch.fnmatch(f, "*.jpg") and os.path.isfile(os.path.join(dir, f)):
                os.remove(os.path.join(dir, f))

    def getCropCount():
        return Film.getFileCount(Film.cropFolder)

    def getScanCount():
        return Film.getFileCount(Film.scanFolder)

    def getRandomFrame(self):
        fileList = Film.getFileList(Film.scanFolder)
        cnt = len(fileList)
        self.scanFileCount = cnt
        if cnt > 0 :
            rn = random.randint(0,cnt-1)
            randomf = os.path.join(Film.scanFolder,fileList[rn])
            self.curFrameNo = rn  
            return Frame(randomf)
        else:
            self.curFrameNo = -1
            return None

    def getLastFrame(self):
        fileList = Film.getFileList(Film.scanFolder)
        cnt = len(fileList)
        self.scanFileCount = cnt
        if cnt > 0 :
            self.curFrameNo = cnt-1
            return Frame(os.path.join(Film.scanFolder,fileList[self.curFrameNo]))
        else:
            self.curFrameNo = -1
            return None 

    def getFirstFrame(self):
        fileList = Film.getFileList(Film.scanFolder)
        cnt = len(fileList)
        self.scanFileCount = cnt
        if cnt > 0 :
            self.curFrameNo = 0
            return Frame(os.path.join(Film.scanFolder,fileList[self.curFrameNo]))
        else:
            self.curFrameNo = -1
            return None 

    def getNextFrame(self):
        fileList = Film.getFileList(Film.scanFolder)
        cnt = len(fileList)
        self.scanFileCount = cnt
        if cnt > 0 :
            if self.curFrameNo+1 >= cnt :
                self.curFrameNo = cnt-1
            elif self.curFrameNo+1 >= 0 :
                self.curFrameNo = self.curFrameNo+1
            else :
                self.curFrameNo = 0
            return Frame(os.path.join(Film.scanFolder,fileList[self.curFrameNo]))
        else:
            self.curFrameNo = -1
            return None   

    def getPreviousFrame(self):
        self.getTargetFrame(self.curFrameNo-1)
        fileList = Film.getFileList(Film.scanFolder)
        cnt = len(fileList)
        self.scanFileCount = cnt
        if cnt > 0 :
            if self.curFrameNo-1 >= cnt :
                self.curFrameNo = cnt-1
            elif self.curFrameNo-1 >= 0 :
                self.curFrameNo = self.curFrameNo-1
            else :
                self.curFrameNo = 0
            return Frame(os.path.join(Film.scanFolder,fileList[self.curFrameNo]))
        else:
            self.curFrameNo = -1
            return None   

    def getTargetFrame(self, target):
        fileList = Film.getFileList(Film.scanFolder)
        cnt = len(fileList)
        self.scanFileCount = cnt
        if cnt > 0 :
            if target >= cnt :
                self.curFrameNo = cnt-1
            elif target >= 0 :
                self.curFrameNo = target
            else :
                self.curFrameNo = 0
            return Frame(os.path.join(Film.scanFolder,fileList[self.curFrameNo]))
        else:
            self.curFrameNo = -1
            return None   

    def cropFrame(self, fileName):
            print(f" instance path {self.scanFolder}")
            print(f" instance path crop {self.cropFolder}")
            print(f"Cropping {fileName} from {self.scanFolder}")
            frame = Frame(os.path.join(self.scanFolder, fileName))
            frame.cropPic()
            outName = fileName.replace("scan","frame")
            cv2.imwrite(os.path.join(self.cropFolder, outName), frame.imageCropped)

    def cropAll(self, progress) :
        frameNo = 0
        os.chdir(Film.scanFolder)
        fileList = sorted(glob.glob('scan*.jpg'))
        self.scanFileCount = len(fileList)
        multi = False
        if multi:
            with ProcessPool(processes=os.cpu_count()) as pool:
                pool.map(self.cropFrame, fileList)
        else:
            for fn in fileList:
                frame = Frame(os.path.join(Film.scanFolder, fn))
                frame.cropPic()
                cv2.imwrite(os.path.join(Film.cropFolder, f"frame{frameNo:06}.jpg"), frame.imageCropped)
                self.curFrameNo = frameNo
                if progress is not None:
                    if progress(frame) == 0:
                        break
                frameNo = frameNo+1
                    
    def makeFilm(self, filmName, progressReport, filmDone) :
        self.progressReport = progressReport
        self.filmDone = filmDone
        os.chdir(Film.cropFolder)
        filmPathName = os.path.join(Film.filmFolder, filmName) + '.mp4'
        if os.path.isfile(filmPathName):
            os.remove(filmPathName)
        
        if self.p is None:  # No process running.
            self.progressReport("Executing process")
            self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            self.p.finished.connect(self.process_finished)  # Clean up once complete.
            self.p.start("ffmpeg", [
                "-framerate", str(self.framerate), 
                "-f", "image2",
                "-pattern_type", "sequence",
                "-i", os.path.join(Film.cropFolder, "frame%06d.jpg"),
                "-s", Film.resolution,
                "-preset", "ultrafast", filmPathName
                ])
   
    def handle_stderr(self):
        data = self.p.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.progressReport(stderr)

    def handle_stdout(self):
        data = self.p.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.progressReport(stdout)

    def handle_state(self, state):
        states = {
            QProcess.NotRunning: 'Not running',
            QProcess.Starting: 'Starting',
            QProcess.Running: 'Running',
        }
        state_name = states[state]
        self.progressReport(f"State changed: {state_name}")

    def process_finished(self):
        self.progressReport("Process finished.")
        self.filmDone()
        self.p = None        


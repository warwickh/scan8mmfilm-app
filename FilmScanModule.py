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
            Frame.outerThresh = config[Ini.frame].getfloat('outerThresh')
            Frame.innerThresh = config[Ini.frame].getfloat('innerThresh')
            Frame.s8_stdSprocketHeight = config[Ini.frame].getfloat('s8_stdSprocketHeight')
            Frame.s8_stdSprocketWidth = config[Ini.frame].getfloat('s8_stdSprocketWidth')
            Frame.s8_ratio = config[Ini.frame].getfloat('s8_ratio')
            Frame.s8_midx = config[Ini.frame].getint('s8_midx')
            Frame.s8_midy = config[Ini.frame].getint('s8_midy')
            Frame.r8_stdSprocketHeight = config[Ini.frame].getfloat('r8_stdSprocketHeight')
            Frame.r8_stdSprocketWidth = config[Ini.frame].getfloat('r8_stdSprocketWidth')
            Frame.r8_ratio = config[Ini.frame].getfloat('r8_ratio')
            Frame.r8_midx = config[Ini.frame].getint('r8_midx')
            Frame.r8_midy = config[Ini.frame].getint('r8_midy')
            Frame.analysisType = config[Ini.frame]['analysisType']
            Frame.s8_frameCrop.load(config)
            #Frame.s8_holeCrop.load(config)
            Frame.r8_frameCrop.load(config)
            #Frame.r8_holeCrop.load(config)
            Frame.s8_whiteCrop.load(config)
            Frame.r8_whiteCrop.load(config)
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
        config[Ini.frame]['outerThresh'] = str(Frame.outerThresh)
        config[Ini.frame]['innerThresh'] = str(Frame.innerThresh)
        config[Ini.frame]['s8_stdSprocketHeight'] = str(Frame.s8_stdSprocketHeight)
        config[Ini.frame]['s8_stdSprocketWidth'] = str(Frame.s8_stdSprocketWidth)
        config[Ini.frame]['s8_ratio'] = str(Frame.s8_ratio)
        config[Ini.frame]['s8_midx'] = str(Frame.s8_midx)
        config[Ini.frame]['s8_midy'] = str(Frame.s8_midy)
        config[Ini.frame]['r8_stdSprocketHeight'] = str(Frame.r8_stdSprocketHeight)
        config[Ini.frame]['r8_stdSprockeWidth'] = str(Frame.r8_stdSprocketWidth)
        config[Ini.frame]['r8_ratio'] = str(Frame.r8_ratio)
        config[Ini.frame]['r8_midx'] = str(Frame.r8_midx)
        config[Ini.frame]['r8_midy'] = str(Frame.r8_midy)
        config[Ini.frame]['analysisType'] = str(Frame.analysisType)

        Frame.s8_frameCrop.save(config)
        #Frame.s8_holeCrop.save(config)
        Frame.s8_whiteCrop.save(config)
        Frame.r8_frameCrop.save(config)
        #Frame.r8_holeCrop.save(config)
        Frame.r8_whiteCrop.save(config)

        with open(inifile, 'w') as configfile:
            config.write(configfile)
        
def getAdjustableRects():
    if Film.format == 's8':
        return [Frame.s8_frameCrop, Frame.s8_whiteCrop]#, Frame.s8_holeCrop]
    else:
        return [Frame.r8_frameCrop, Frame.r8_whiteCrop]#, Frame.r8_holeCrop]
    
def getAnalysisTypes():
    return ["auto", "thresh", "ratio", "manual"]

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
        if config.has_section(self.name):
            self.x1 = config[self.name].getint('x1')
            self.x2 = config[self.name].getint('x2')
            self.y1 = config[self.name].getint('y1')
            self.y2 = config[self.name].getint('y2')
        else:
            print(f"Config for {self.name} doesn't exist")

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
        self.x1 = self.x1 + adj
        self.x2 = self.x2 + adj

    def adjY(self, adj):
        #if self.y1 + adj >= 0 :
        self.y1 = self.y1 + adj
        self.y2 = self.y2 + adj
        
    def adjXSize(self, adj):
        if self.x2 + adj > self.x1 :
            self.x2 = self.x2 + adj
    
    def adjYSize(self, adj):
        if self.y2 + adj > self.y1 :
            self.y2 = self.y2 + adj 

class Frame:

    format = "s8"
    s8_frameCrop = Rect("s8_frame_crop", 2, -107, 2+1453, 1040-107)
    #s8_holeCrop = Rect("s8_hole_crop", 75, 0, 110, 2463) 
    #s8_holeCrop = Rect("s8_hole_crop", 385, 0, 564, 2463) 
    s8_whiteCrop = Rect("s8_white_crop", -185, -109, -27, 122)
    s8_stdSprocketHeight = 0.13
    s8_stdSprocketWidth = 0.055
    s8_midx = 64
    s8_midy = 240
    r8_frameCrop = Rect("r8_frame_crop", 146, 28, 146+814, 28+565)
    #r8_holeCrop = Rect("r8_hole_crop", 75, 0, 240, 276)
    #r8_holeCrop = Rect("r8_hole_crop", 385, 0, 1230, 1415)
    r8_whiteCrop = Rect("r8_white_crop",  436, 1078, 631, 1362)
    r8_stdSprocketHeight = 0.1
    r8_stdSprocketWidth = 0.13    
    r8_midx = 64
    r8_midy = 120
    ScaleFactor = 1.0 # overwritten by initScaleFactor()
    outerThresh = 0.6
    innerThresh = 0.3
    s8_ratio = (4.23-1.143)/1.143 #gap/sprocket width
    r8_ratio = (4.23-1.27)/1.143 #gap/sprocket width
    hist_path = os.path.expanduser("~/my_cv2hist_lim.png")
    whiteThreshold = 225
    analysisType = 'auto'
    ratioX1 = 385
    ratioX2 = 564

    def initScaleFactor():
        Frame.ScaleFactor = Camera.ViewWidth/640.0 
              
    def getHoleCropWidth():
        return Frame.ratioX2-Frame.ratioX1
        #if Frame.format == "s8":
        #    return 1*(Frame.s8_holeCrop.x2-Frame.s8_holeCrop.x1) #wider to capture vertical line also
        #else:
        #    return 1*(Frame.r8_holeCrop.x2-Frame.r8_holeCrop.x1) #wider to capture vertical line also
        #return self.holeCrop.x2 - Frame.holeCrop.x1
    
    def __init__(self, imagePathName=None,*,image=None):
        self.imagePathName = imagePathName
        if image is None and imagePathName is not None :
            self.imagePathName = imagePathName
            self.image = cv2.imread(imagePathName)
            #self.threshImgPath = os.path.join(os.path.dirname(self.imagePathName),"whitethresh.png")
        elif image is not None :
            self.image = image
            #self.threshImgPath = "whitethresh.png"
        self.dy,self.dx,_ = self.image.shape
        self.ScaleFactor = self.dx/640.0
        print(f"Scalefactor {Frame.ScaleFactor}")
        if Frame.format == "s8":
            print(f"Checking {Frame.s8_stdSprocketWidth}")
            print(f"Checking {Frame.s8_stdSprocketHeight}")
            self.stdSprocketHeight = Frame.s8_stdSprocketHeight*self.dy
            self.stdSprocketWidth = Frame.s8_stdSprocketWidth*self.dx          
            #self.holeCrop = Frame.s8_holeCrop#Rect("hole_crop", Frame.s8_holeCrop.x1*self.ScaleFactor, 0, Frame.s8_holeCrop.x2*self.ScaleFactor, self.dy-1)
            self.frameCrop = Frame.s8_frameCrop
            self.whiteCrop = Frame.s8_whiteCrop
            self.ratio = Frame.s8_ratio
            self.midy = Frame.s8_midy*self.ScaleFactor
        else:
            self.stdSprocketHeight = Frame.r8_stdSprocketHeight*self.dy
            self.stdSprocketWidth = Frame.r8_stdSprocketWidth*self.dx
            #self.holeCrop = Frame.r8_holeCrop#Rect("hole_crop", Frame.r8_holeCrop.x1*self.ScaleFactor, 0, Frame.r8_holeCrop.x2*self.ScaleFactor, self.dy-1)
            self.frameCrop = Frame.r8_frameCrop
            self.whiteCrop = Frame.r8_whiteCrop
            self.ratio = Frame.r8_ratio
            self.midy = Frame.r8_midy*self.ScaleFactor
        self.minSprocketHeight = self.stdSprocketHeight*0.7
        self.maxSprocketHeight = self.stdSprocketHeight*1.3
        print(f"Sprocket min {self.minSprocketHeight} max {self.maxSprocketHeight}")
        self.midx = 115*self.ScaleFactor   # always overwitten 
        #self.midy = self.dy//2#240*self.ScaleFactor 
        self.cX = self.midx 
        self.cY = self.midy
        self.rX = self.cX + self.stdSprocketWidth
        self.lX = 300
        self.filmEdge = 0
        self.sprocketSize = 0    
        self.histogram = None
        #self.histogram = cv2.imread(os.path.expanduser("~/my_cv2hist_lim.png"))
        self.locateHoleResult = 1
        #print(f"init complete {self.__dict__}")
        #self.whiteThreshold = self.getWhiteThreshold(self.threshImgPath)
        #self.analysisType = Frame.analysisType
        
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
        
    def getQPixmap(self, dest=None):
        return self.convert_cv_qt(self.image, dest)
        
    def getCropped(self, dest=None):
        ScaleFactor = int(self.ScaleFactor)
        outCrop = cv2.resize(self.imageCropped, (0,0), fx=1/ScaleFactor, fy=1/ScaleFactor)
        return self.convert_cv_qt(outCrop, dest)
        
    def getHoleCrop(self) :
        #cv2.imwrite(os.path.expanduser("~/getHoleCrop.png"), self.imageHoleCrop)
        #self.imageHoleCrop = cv2.resize(cv2.imread(self.resultImagePath), (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)
        self.imageHoleCrop = cv2.resize(self.imageHoleCrop, (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)
        return self.convert_cv_qt(self.imageHoleCrop)

    def getHistogram(self):
        width = 200
        self.histogram = cv2.imread(Frame.hist_path)
        y,x,_ = self.histogram.shape
        scale = 200/y
        #self.histogram = cv2.resize(self.histogram, (200, 200))
        return self.convert_cv_qt(cv2.resize(self.histogram, (0,0), fx=scale, fy=scale))

    def calcCrop(self):
        #self.locateHoleResult = self.locateSprocketHole(Frame.holeMinArea)
        self.locateHoleResult = self.locateSprocketHole()
        #x = int((self.cX + Frame.holeCrop.x1) * Frame.ScaleFactor)+Frame.frameCrop.x1
        #y = int((self.cY + Frame.holeCrop.y1) * Frame.ScaleFactor)+Frame.frameCrop.y1 
        x = int(self.rX + (self.frameCrop.x1 * Frame.ScaleFactor))
        y = int(self.cY + (self.frameCrop.y1 * Frame.ScaleFactor)) 
        self.p1 = (x, y)
        self.p2 = (x+self.frameCrop.getXSize(), y+self.frameCrop.getYSize())
        print(f"crop self.p1 {self.p1} self.p2 {self.p2}")
        
    #def getWhiteOutline(self, dest=None):
    #    wp1 = (round(Frame.whiteCrop.x1 * Frame.ScaleFactor), round(Frame.whiteCrop.y1 * Frame.ScaleFactor))
    #    wp2 = (round(Frame.whiteCrop.x2 * Frame.ScaleFactor), round(Frame.whiteCrop.y2 * Frame.ScaleFactor))
    #    cv2.rectangle(self.image, wp1, wp2, (60, 240, 240), 10)
    #    return self.convert_cv_qt(self.image, dest)
    
    def getCropOutline(self, dest=None):
        self.calcCrop()
        cv2.rectangle(self.image, self.p1, self.p2, (0, 255, 0), 10)
        wp1 = (round(self.rX+self.whiteCrop.x1), round(self.cY+self.whiteCrop.y1))
        wp2 = (round(self.rX+self.whiteCrop.x2), round(self.cY+self.whiteCrop.y2))
        cv2.rectangle(self.image, wp1, wp2, (60, 240, 240), 10)
        return self.convert_cv_qt(self.image, dest)
        
    def cropPic(self):
        self.calcCrop()
        self.imageCropped = self.image[self.p1[1]:self.p2[1], self.p1[0]:self.p2[0]]
     
    def getWhiteThreshold(self, threshFilename=None):
        if threshFilename:
            print(f"Checking for threshold file at {threshFilename} {os.path.exists(threshFilename)}")
            img = cv2.imread(threshFilename)
        else:
            #img = self.image[Frame.whiteCrop.y1:Frame.whiteCrop.y2, Frame.whiteCrop.x1:Frame.whiteCrop.x2]
            img = self.image[self.cY+self.whiteCrop.y1:self.cY+self.whiteCrop.y2, self.rX+self.whiteCrop.x1:self.rX+self.whiteCrop.x2]
        dy,dx,_ = img.shape
        #img = imageSmall[self.whiteCrop.y1:self.whiteCrop.y2, self.whiteCrop.x1:self.whiteCrop.x2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        planes = cv2.split(img)
        histSize = 256 #  [Establish the number of bins]
        histRange = (0, 256) # Set the range
        hist = cv2.calcHist(planes, [0], None, [histSize], histRange, accumulate=False) 
        okPct = dy*dx/100.0*5
        #okPct = (self.whiteCrop.y2-self.whiteCrop.y1)*(self.whiteCrop.x2-self.whiteCrop.x1)/100.0*5
        wco = 220 # Default value - will work in most cases
        for i in range(128,256) :
            if hist[i] > okPct :
                wco = i-8 #6
                break
        print(f"Found threshold {wco} from {threshFilename}")
        return wco     

    def locateSprocketHole(self):
        print(f"Locating sprocket hole using {Frame.analysisType}")
        if Frame.analysisType=='auto' or Frame.analysisType=='thresh':
            locateHoleResult = self.locateSprocketHoleThresh()#Frame.holeMinArea)
            print(f"Thresh cY {self.cY} rX {self.rX} locateHoleResult {locateHoleResult}")
        elif (Frame.analysisType=='auto' and locateHoleResult != 0) or Frame.analysisType=='ratio' :
            locateHoleResult = self.locateSprocketHoleRatio()#Frame.holeMinArea)
            print(f"Ratio cY {self.cY} rX {self.rX} locateHoleResult {locateHoleResult}")
        self.locateHoleResult = locateHoleResult
        if not locateHoleResult==0:
            raise Exception(f"Nonzero locateHoleResult {locateHoleResult}")

    # Based on https://github.com/cpixip/sprocket_detection
    # initial testing had issues with selecting point between 2 holes TODO
    # Added initial range check
    # Return values:
    # 0: hole found, 1: hole not found, 2: hole to large, 3: no center
    def locateSprocketHoleRatioOld(self):
        print(f"locateSprocketHole {self.image.shape}")
        #thresholds = [0.5,0.3]          # edge thresholds; first one higher, second one lower
        filterSize = 25                 # smoothing kernel - leave it untouched
        #print(f"Check Frame.holeCrop.y1 {Frame.holeCrop.y1} Frame.holeCrop.y2 {Frame.holeCrop.y2}")
        midy = self.midy
        dy = self.dy
        y1 = 0
        y2 = self.dy
        #x1 = int(Frame.holeCrop.x1*Frame.ScaleFactor)
        #x2 = int(Frame.holeCrop.x2*Frame.ScaleFactor)
        x1 = int(Frame.ratioX1)
        x2 = int(Frame.ratioX2)
        print(f"x1 {x1} x2 {x2} y1 {y1} y2 {y2}")
        self.imageHoleCrop = self.image[:,int(x1):int(x1+2*(x2-x1)),:]
        self.imageHoleCropHide = self.image[:,int(x1):int(x2),:]
        sprocketEdges = np.absolute(cv2.Sobel(self.imageHoleCropHide,cv2.CV_64F,0,1,ksize=3))
        histogram     = np.mean(sprocketEdges,axis=(1,2))
        smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)
        maxPeakValue   = smoothedHisto[y1:y2].max()
        minPeakValue   = smoothedHisto[y1:y2].min()
        outerThreshold = Frame.outerThresh*maxPeakValue
        innerThreshold = Frame.innerThresh*maxPeakValue       
        outerLow = y1
        #thresh_vals = [outerThreshold+10, outerThreshold+5, outerThreshold, outerThreshold-5, outerThreshold-10]
        thresh_vals = [outerThreshold, outerThreshold-5, outerThreshold+5]
        print(f"Thresh vals {thresh_vals} maxPeakValue {maxPeakValue}")
        for z in thresh_vals:
            peaks = []
            trough = None
            for y in range(y1,y2):
                if smoothedHisto[y]<z and smoothedHisto[y+1]>z:
                    peaks.append(y)
                if smoothedHisto[y]==minPeakValue:
                    trough=y
            print(f"Peaks at {z:.2f} {peaks} thresh {outerThreshold:.2f}")
            if len(peaks)>2 and len(peaks)<5:
                print(f"Got enough peaks {peaks} {len(peaks)} at {z:.2f} midy {midy} thresh {outerThreshold:.2f} trough at {trough}")
                break
        #print(f"Trough {trough}")
        #Find a range containing a sprocket closest to centre by comparing sprocket/not sprocket gaps with ratio
        #detected sprocket must be within 0.3 of the frame
        #frameLocated=False
        sprocketStartList=[]
        for i in range(0,len(peaks)-2):
            print(f"Ratio {self.ratio}")
            #print(f"Ratio {(peaks[i+1]-peaks[i])/(peaks[i+2]-peaks[i+1])} {(peaks[i+2]-peaks[i+1])/(peaks[i+1]-peaks[i])} ideal {ideal_ratio}")
            if (((peaks[i+1]-peaks[i])/(peaks[i+2]-peaks[i+1]) - self.ratio) < 0.5):
                sprocketStartList.append(peaks[i+1])
                print(f"Ratio Back {((peaks[i+1]-peaks[i])/(peaks[i+2]-peaks[i+1]))}")
            elif (((peaks[i+2]-peaks[i+1])/(peaks[i+1]-peaks[i]) - self.ratio) < 0.5):
                sprocketStartList.append(peaks[i])
                print(f"Ratio forward {((peaks[i+2]-peaks[i+1])/(peaks[i+1]-peaks[i]))}")
            #print(f"Found sprocket at {sprocketStart} using ratio which is {(midy-sprocketStart)/dy:.2f} from the centre")
        minDist = 1
        sprocketStart = 0
        for sprocket in sprocketStartList:
            dist = abs((midy-sprocket)/dy)
            print(f"{sprocket} {dist}")
            if dist<minDist:
                minDist=dist
                sprocketStart = sprocket
        print(f"Best sprocket at {sprocketStart} with {minDist}")
        if minDist<0.3:
            y1=int(sprocketStart-(1.5*self.maxSprocketHeight))
            y2=int(sprocketStart+(1.5*self.maxSprocketHeight))
            print(f" Sprocket within range, so new y1 {y1} new y2 {y2}")
            frameLocated = True
            #break
        #Locate sprocket in reduced range
        for y in range(y1,y2):
            if smoothedHisto[y]>outerThreshold:
                outerLow = y                 
                break
        outerHigh      = y2
        for y in range(y2,outerLow,-1):
            if smoothedHisto[y]>outerThreshold:
                outerHigh = y
                break
        print(f"outerHigh {outerHigh} outerLow {outerLow} outerHigh-outerLow {outerHigh-outerLow}") #Might need to do some cleanup here for redundancy
        if (outerHigh-outerLow)<0.3*dy:
            searchCenter = (outerHigh+outerLow)//2
        else:
            #searchCenter = dy//2
            #searchCenter = int(outerLow + (0.5*self.minSprocketHeight)) #give priority to top frame. Try to find internal location - probably can change to outerhigh-outerlow
            #searchCenter = int(outerHigh - (0.5*self.minSprocketHeight)) #give priority to top frame. Try to find internal location - probably can change to outerhigh-outerlow
            searchCenter = int(trough)
            print(f"Between 2 frames, using trough. Searching from {searchCenter}")
        innerLow = searchCenter
        for y in range(searchCenter,outerLow,-1):
            if smoothedHisto[y]>innerThreshold:
                innerLow = y
                break
        innerHigh = searchCenter
        for y in range(searchCenter,outerHigh):
            if smoothedHisto[y]>innerThreshold:
                innerHigh = y
                break
        sprocketSize    = innerHigh-innerLow
        #minSprocketHeight = int(minSize)
        #print(f"minSprocketHeight {self.minSprocketHeight}<sprocketSize {sprocketSize} {self.minSprocketHeight<sprocketSize}")
        #print(f"maxSprocketHeight {self.maxSprocketHeight}>sprocketSize {sprocketSize} {self.maxSprocketHeight>sprocketSize}")
        #print(f"and sprocketSize<(outerHigh-outerLow) {sprocketSize<(outerHigh-outerLow)}")
        if self.minSprocketHeight<sprocketSize and sprocketSize<(outerHigh-outerLow) and sprocketSize<self.maxSprocketHeight:
            cY = (innerHigh+innerLow)//2
            print(f"Valid sprocket size {sprocketSize}")
            locateHoleResult = 0
        elif sprocketSize>self.maxSprocketHeight:
            cY = (innerHigh+innerLow)//2
            print(f"Invalid sprocket size too big {sprocketSize}")
            locateHoleResult = 2
        else:
            print(f"Why am I here with sprocketSize {sprocketSize}")
            print(f"probably not enough peaks found {len(peaks)}")
            cY = dy//2
            sprocketSize   = 0
            locateHoleResult = 1
        cX = x1
        locatedX = False
        if sprocketSize>0:
            rx1 = x1
            rx2 = x1 + 4*(x2-x1)
            ry = int(0.8*sprocketSize)
            ry1 = cY-ry//2
            ry2 = cY+ry//2
            horizontalStrip = self.image[int(ry1):int(ry2),int(rx1):int(rx2),:]
            horizontalEdges = np.absolute(cv2.Sobel(horizontalStrip,cv2.CV_64F,1,0,ksize=3))
            histoHori       = np.mean(horizontalEdges,axis=(0,2))
            smoothedHori    = cv2.GaussianBlur(histoHori,(1,5),0)
            maxPeakValueH   = smoothedHori.max()
            thresholdHori   = Frame.innerThresh*maxPeakValueH
            for x in range((x2-x1)//2,len(smoothedHori)):
                if smoothedHori[x]>thresholdHori:
                    #xShift = x                 
                    cX = x+x1
                    locatedX = True                 
                    break
        oldcX = self.cX
        oldcY = self.cY  
        if locatedX:
            self.cX = cX
        self.cY = cY
        #locateHoleResult = 0
        print(f"InnerLow {innerLow} InnerHigh {innerHigh} cY {cY} cX {cX}")
        print(f"Found sprocket edge {locatedX} at {cX}")
        print("cY=", self.cY, "oldcY=", oldcY, "locateHoleResult=", locateHoleResult)
 
 
        p1 = (int(x2-x1), 0) 
        p2 = (int(x2-x1), dy-1) 
        print(f"Vertical line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (255, 0, 0), 3) #View X1 Blue
 
 
        p1 = (0, int(self.cY))
        p2 = (int(self.cX-x1), int(self.cY))
        print(f"Horizontal line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Horiz
        p1 = (int(self.cX-x1), int(y1)) 
        p2 = (int(self.cX-x1), int(y2)) 
        print(f"Vertical line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Vert
        #cv2.line(self.image, p1, p2, (0, 0, 255), 3) #Vert
        # show target midy
        p1 = (0, int(midy)) 
        p2 = (int(self.cX-x1), int(midy))
        print(f"MidY line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 0), 3)  # black line
        p1 = (0, int(midy+3))
        p2 = (int(self.cX-x1), int(midy+3))
        cv2.line(self.imageHoleCrop, p1, p2, (255, 255, 255), 3) # white line
        
        self.imageHoleCrop = cv2.resize(self.imageHoleCrop, (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)

        plt.plot(smoothedHisto)
        plt.axvline(cY, color='blue', linewidth=1)
        plt.axvline(searchCenter, color='orange', linewidth=1)
        plt.axvline(innerHigh, color='green', linewidth=1)
        plt.axvline(innerLow, color='red', linewidth=1)
        plt.axvline(outerHigh, color='purple', linewidth=1)
        plt.axvline(outerLow, color='gray', linewidth=1)
        plt.axhline(innerThreshold, color='cyan', linewidth=1)
        plt.axhline(outerThreshold, color='olive', linewidth=1)
        plt.axvline(trough, color='pink', linewidth=1)
        plt.xlim([0, dy])
        #plt.show()
        plt.savefig(os.path.expanduser("~/my_cv2hist_full.png"))
        plt.xlim(y1,y2)
        plt.savefig(os.path.expanduser("~/my_cv2hist.png"))
        #plt.show()
        #plt.savefig(Frame.hist_path)#os.path.expanduser("~/my_cv2hist_lim.png"))
        #self.histogram = cv2.imread(os.path.expanduser("~/my_cv2hist_lim.png"))
        plt.clf()
        cv2.imwrite(os.path.expanduser("~/sprocketStrip.png"), self.imageHoleCrop)
        cv2.imwrite(os.path.expanduser("~/image.png"), self.image)
        if locatedX:
            cv2.imwrite(os.path.expanduser("~/horizontalStrip.png"), horizontalStrip)
           
        #self.locateHoleResult = locateHoleResult
        return locateHoleResult

#=========================================================================
        
    def findSprocket(self, x1, x2):
        y1=0
        y2=self.dy-1
        if len(self.smoothedHisto)==0:
            return None, None
        maxPeakValue   = self.smoothedHisto[y1:y2].max()
        minPeakValue   = self.smoothedHisto[y1:y2].min()
        #print(f"maxPeakValue {maxPeakValue}")
        if maxPeakValue<50:
            print("Not enough range in hist -------------------------------------------------")
        plt.plot(self.smoothedHisto)
        sprocketStart = None
        sprocketHeight = None
        for z in range(int(maxPeakValue*0.8),int(minPeakValue),int(-0.1*(maxPeakValue-minPeakValue))):
            plt.axhline(z, color='blue', linewidth=1)
            peaks = []
            for y in range(y1,y2):
                if self.smoothedHisto[y]<z and self.smoothedHisto[y+1]>z:
                    peaks.append(y)
                    if len(peaks)>1:
                        print(f"testing for match with peak list {peaks} at {z}")
                        for peak in peaks:
                            testPeaks = [peak+self.stdSprocketHeight
                                ,peak-self.stdSprocketHeight
                                ,peak+self.stdSprocketHeight*self.ratio*1.02
                                ,peak-self.stdSprocketHeight*self.ratio*1.02]
                            #print(f"testpeaks {testPeaks}")
                            valueSet = [peak]
                            sprocketStart = None
                            for i in range(len(testPeaks)):
                            #for testPeak in testPeaks:
                                upper = testPeaks[i]*1.1
                                lower = testPeaks[i]*0.9
                                #plt.axvline(testPeak, color='red', linewidth=1,label=f"testPeak")
                                for loc in peaks:
                                    if lower<loc<upper:
                                        print(f"peak {peak} testPeak {i} {testPeaks[i]} Found loc {loc} within {lower} {upper} so good")
                                        if i==0:
                                            print(f"i is 0 so {peak} should be the sprocketstart")
                                            sprocketStart = peak
                                        elif i==1:
                                            print(f"i is 1 so the peak before {peak} should be the sprocketstart")
                                            #print(f"try {peaks[peaks.index(peak)-1]}")
                                            sprocketStart = peaks[peaks.index(peak)-1]
                                        plt.axvline(loc, color='green', linewidth=1,label=f"yes")
                                        valueSet.append(loc)
                                    else:
                                        print(f"Failed peak {peak} testPeak {i} {testPeaks[i]} loc {loc} not within {lower} {upper} so fail")
                                        
                            #print(f"End of test with peak list found {len(valueSet)} values {valueSet} from peaks {peaks}")            
                            if len(valueSet)>1 and sprocketStart:
                                valueSet.sort()   
                                print(f"Got enough matching vals at {valueSet} sprocketstart {sprocketStart}")
                                sprocketEnd = valueSet[valueSet.index(sprocketStart)+1]#TODO crashing here when peak is too low ValueError: 14 is not in list
                                sprocketHeight = sprocketEnd - sprocketStart
                                sprocketCentre = sprocketStart+0.5*sprocketHeight
                                #print(f"sprocketCentre {sprocketCentre}")
                                dist = abs((self.midy-sprocketCentre)/self.dy)
                                print(f"{valueSet} sprocketstart {sprocketStart} sprocketEnd {sprocketEnd} sprocketSize {sprocketHeight} dist {dist:02f}")  
                                if dist<0.3:
                                    print(f"Found sprocket in range at {valueSet}")
                                    plt.savefig(os.path.expanduser("~/findsprocket.png"))
                                    return sprocketStart, sprocketHeight
                    plt.savefig(os.path.expanduser(f"~/findsprocketfail_{y}.png"))                        
        return None, None


    def findYRange(self, x1, x2):
        y1=0
        y2=self.dy-1
        returnY1 = None
        returnY2 = None
        if len(self.smoothedHisto)==0:
            return None, None
        maxPeakValue   = self.smoothedHisto[y1:y2].max()
        minPeakValue   = self.smoothedHisto[y1:y2].min()
        print(f"maxPeakValue {maxPeakValue}")
        fullrange = maxPeakValue-minPeakValue
        if maxPeakValue<50:
            print("Not enough range in hist -------------------------------------------------")
        self.outerThreshold = Frame.outerThresh*maxPeakValue
        self.innerThreshold = Frame.innerThresh*maxPeakValue
        plt.plot(self.smoothedHisto)
        #thresh_vals = [outerThreshold+10, outerThreshold+5, outerThreshold, outerThreshold-5, outerThreshold-10]
        #thresh_vals = [self.outerThreshold, self.outerThreshold-5, self.outerThreshold-10,self.outerThreshold+5]
        #thresh_vals = [self.outerThreshold, self.outerThreshold*0.9, self.outerThreshold*0.8,self.outerThreshold*0.7, self.outerThreshold*0.6,self.outerThreshold*0.5]
        #print(f"Thresh vals {thresh_vals} Frame.outerThresh {Frame.outerThresh}")
        finalZ = 0
        thresh_vals = [self.outerThreshold]
        #for z in range(int(maxPeakValue),int(minPeakValue),int(-0.1*fullrange)):
        for z in thresh_vals:
            plt.axhline(z, color='blue', linewidth=1)
            peaks = []
            #trough = None
            #count=0
            #valueSets=[]
            for y in range(y1,y2):
                if self.smoothedHisto[y]<z and self.smoothedHisto[y+1]>z:
                    peaks.append(y)
                    #plt.clf() #Clear graph
                    #plt.plot(self.smoothedHisto)
        
                    print(f"Found a new peak {y}, testing for match with peak list{peaks}")
                    for peak in peaks:
                        testPeaks = [peak+self.stdSprocketHeight
                            ,peak-self.stdSprocketHeight
                            ,peak+self.stdSprocketHeight*self.ratio*1.02
                            ,peak-self.stdSprocketHeight*self.ratio*1.02]
                        #print(f"peak {peak} self.stdSprocketHeight {self.stdSprocketHeight} peak+self.stdSprocketHeight*self.ratio {peak+self.stdSprocketHeight*self.ratio}")
                        #plt.axvline(testPeaks[0], color='orange', linewidth=1,label=f"tp1")
                        #plt.axvline(testPeaks[1], color='green', linewidth=1,label=f"tp2")
                        #plt.axvline(testPeaks[2], color='blue', linewidth=1,label=f"tp3")
                        #plt.axvline(testPeaks[3], color='cyan', linewidth=1,label=f"tp4")
                        #plt.axvline(peak, color='red', linewidth=1,label=f"found")
                        #plt.legend(loc=0)
                        #print(f"testpeaks {testPeaks}")
                        valueSet = [peak]
                        for i in range(len(testPeaks)):
                        #for testPeak in testPeaks:
                            upper = testPeaks[i]*1.1
                            lower = testPeaks[i]*0.9
                            #plt.axvline(testPeak, color='red', linewidth=1,label=f"testPeak")
                            for loc in peaks:
                                if lower<loc<upper:
                                    #print(f"peak {peak} testPeak {i} {testPeaks[i]} Found {loc} within {lower} {upper} so good")
                                    if i==0:
                                        print(f"i is 0 so {peak} should be the sprocketstart")
                                    elif i==1:
                                        print(f"i is 1 so the peak before {peak} should be the sprocketstart")
                                        print(f"try {peaks[peaks.index(peak)-1]}")
                                    plt.axvline(loc, color='green', linewidth=1,label=f"yes")
                                    valueSet.append(loc)
                                #else:
                                    #print(f"peak {peak} testPeak {testPeak} Item {loc} not in range {lower} {upper} for {testPeak} so fail")
                                    #plt.axvline(loc, color='red', linewidth=1,label=f"no")
                            #valueSets.append(valueSet)
                        print(f"{valueSet}")  
                        if len(valueSet)>2:
                            valueSet.sort()
                            print(f"Found sprocket at {valueSet}")
                            for i in range(len(valueSet)-1):
                                testVal = valueSet[i+1]/self.ratio
                                print(f"Testing {valueSet[i]} against {valueSet[i+1]} {testVal}")
                            break




                    #count+=1
                    #if count>2:
                    #    break
            #print(valueSets)          

                #if self.smoothedHisto[y]==minPeakValue:
                #    self.trough=y
            #print(f"Peaks at {z:.2f} {peaks} thresh {self.outerThreshold:.2f}")
            #closePeaks = 0
            #for peak in peaks:
            #    if abs(peak-self.midy)/self.dy<0.3:
            #        print(f"Found a close peak at peak {peak}")
            #        closePeaks+=1
            #if len(peaks)>2 and len(peaks)<6 and finalZ==0 and closePeaks>1:
            #    self.peaks=peaks
            #    finalZ = z
            #    print(f"Got enough peaks {self.peaks} {len(self.peaks)} at {z:.2f} midy {self.midy} thresh {self.outerThreshold:.2f} trough at {self.trough}")
            #    break
            #for i in self.peaks:
            #    plt.axvline(i, color='blue', linewidth=1)


        #print(f"Found sprocket at {valueSet}")
        plt.xlim([0, self.dy])
        plt.savefig(os.path.expanduser("~/my_cv2histb.png"))
        plt.clf()
        cv2.imwrite(os.path.expanduser("~/b.png"), self.image)
        cv2.imwrite(os.path.expanduser("~/threshb.png"), self.thresh)
        if finalZ==0:
            print(f"Not enough peaks")
            #raise Exception(f"Not enough peaks {self.imagePathName}")
            return 0,0
        #print(f"Trough {trough}")
        #Find a range containing a sprocket closest to centre by comparing sprocket/not sprocket gaps with ratio
        #detected sprocket must be within 0.3 of the frame
        frameLocated=False
        sprocketStart=None
        for i in range(0,len(self.peaks)-2):
            print(f"Ratio {self.ratio}")
            #print(f"Ratio {(self.peaks[i+1]-self.peaks[i])/(self.peaks[i+2]-self.peaks[i+1])} {(self.peaks[i+2]-self.peaks[i+1])/(self.peaks[i+1]-self.peaks[i])} ideal {ideal_ratio}")
            if (((self.peaks[i+1]-self.peaks[i])/(self.peaks[i+2]-self.peaks[i+1]) - self.ratio) < 0.5):
                sprocketStart = self.peaks[i+1]
                print(f"Ratio Back {((self.peaks[i+1]-self.peaks[i])/(self.peaks[i+2]-self.peaks[i+1]))}")
            elif (((self.peaks[i+2]-self.peaks[i+1])/(self.peaks[i+1]-self.peaks[i]) - self.ratio) < 0.5):
                sprocketStart = self.peaks[i]
                print(f"Ratio forward {((self.peaks[i+2]-self.peaks[i+1])/(self.peaks[i+1]-self.peaks[i]))}")
            print(f"Found sprocket at {sprocketStart} using ratio which is {(self.midy-sprocketStart)/self.dy:.2f} from the centre")
            if sprocketStart and abs((self.midy-sprocketStart)/self.dy)<0.3:
                returnY1=int(sprocketStart-(1*self.maxSprocketHeight))
                returnY2=int(sprocketStart+(1.5*self.maxSprocketHeight))
                print(f" Sprocket within range, so new y1 {returnY1} new y2 {returnY2}")
                frameLocated = True
                break
        if frameLocated:
            Frame.outerThresh = finalZ/maxPeakValue
            self.outerThreshold = finalZ
            print(f"Setting Frame.outerThresh to {finalZ} saved as {Frame.outerThresh}")
        return returnY1, returnY2

    def findSprocketSize(self, y1, y2):
        print(f"Find sprocket size {y1} {y2}")
        if not y1 or not y2:
            return 0,0
        outerLow = y1
        for y in range(y1,y2):
            if self.smoothedHisto[y]>self.outerThreshold:
                outerLow = y                 
                break
        outerHigh      = y2
        for y in range(y2,outerLow,-1):
            if self.smoothedHisto[y]>self.outerThreshold:
                outerHigh = y
                break
        print(f"outerHigh {outerHigh} outerLow {outerLow} outerHigh-outerLow {outerHigh-outerLow}") #Might need to do some cleanup here for redundancy
        if (outerHigh-outerLow)<0.3*self.dy:
            searchCenter = (outerHigh+outerLow)//2
        else:
            searchCenter = int(self.trough)
            print(f"Between 2 frames, using trough. Searching from {searchCenter}")
        innerLow = searchCenter
        for y in range(searchCenter,outerLow,-1):
            if self.smoothedHisto[y]>self.innerThreshold:
                innerLow = y
                break
        innerHigh = searchCenter
        for y in range(searchCenter,outerHigh):
            if self.smoothedHisto[y]>self.innerThreshold:
                innerHigh = y
                break
        sprocketHeight    = innerHigh-innerLow
        cY = (innerHigh+innerLow)//2
        plt.plot(self.smoothedHisto)
        plt.axvline(cY, color='blue', linewidth=1)
        plt.axvline(searchCenter, color='orange', linewidth=1)
        plt.axvline(innerHigh, color='green', linewidth=1)
        plt.axvline(innerLow, color='red', linewidth=1)
        plt.axvline(outerHigh, color='purple', linewidth=1)
        plt.axvline(outerLow, color='gray', linewidth=1)
        plt.axhline(self.innerThreshold, color='cyan', linewidth=1)
        plt.axhline(self.outerThreshold, color='olive', linewidth=1)
        plt.axvline(self.trough, color='pink', linewidth=1)
        plt.xlim([0, self.dy])
        #plt.show()
        plt.savefig(os.path.expanduser("~/my_cv2hist.png"))
        #plt.show()
        plt.xlim(y1,y2)
        plt.savefig(self.hist_path)#os.path.expanduser("~/my_cv2hist_lim.png"))
        #self.histogram = cv2.imread(os.path.expanduser("~/my_cv2hist_lim.png"))
        plt.clf()
        return cY, sprocketHeight
    
    def locateSprocketHoleRatio(self):
        filterSize = 25                 # smoothing kernel - leave it untouched
        midy = self.midy
        self.peaks = []
        self.smoothedHisto = []
        lX = self.findSprocketLeft()
        x1 = int(lX+(0.01*self.dx))#buffer for rough edge
        if not x1:
            cv2.imwrite(os.path.expanduser("~/ratioa.png"), self.image)
            cv2.imwrite(os.path.expanduser("~/ratiothresh.png"), self.thresh)
            locateHoleResult = 5 #can't find left edge
            print(f"Setting result to 5 - can't find left edge {self.imagePathName}")
            raise Exception(f"Setting result to 5 - can't find left edge {self.imagePathName}")
    
        x2 = x1+int(self.stdSprocketWidth*0.8)
        self.imageHoleCrop = self.image[:,int(x1):int(x1+2*(x2-x1)),:]
        self.imageHoleCropHide = self.image[:,int(x1):int(x2),:]
        hsvMargin=50
        image = cv2.cvtColor(self.imageHoleCropHide, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 255-hsvMargin], dtype="uint8")
        upper = np.array([255, hsvMargin, 255], dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        masked = cv2.bitwise_and(self.imageHoleCropHide, self.imageHoleCropHide, mask=mask)
        cv2.imwrite(os.path.expanduser("~/whitemsk.png"), mask)
        cv2.imwrite(os.path.expanduser("~/masked.png"), masked)
        cv2.imwrite(os.path.expanduser("~/imageHoleCropHide.png"), self.imageHoleCropHide)
        #sprocketEdges = np.absolute(cv2.Sobel(self.imageHoleCropHide,cv2.CV_64F,0,1,ksize=3))
        sprocketEdges = np.absolute(cv2.Sobel(mask,cv2.CV_64F,0,1,ksize=3))
        #histogram     = np.mean(sprocketEdges,axis=(1,2))
        histogram     = np.mean(sprocketEdges,axis=(1))
        self.smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)
        Frame.ratioX1 = x1
        Frame.ratioX2 = x2
        #Find the y search range
        sprocketStart, sprocketHeight = self.findSprocket(x1,x2)
        print(f"Processing {sprocketStart} {sprocketHeight}")
        if not sprocketStart or not sprocketHeight:
            locateHoleResult = 4 #Cant find sprocket/gap pattern
            print("Setting result to 4 - can't match pattern")
        else:
            cY = sprocketStart+0.5*sprocketHeight
            self.sprocketHeight = sprocketHeight
            if self.minSprocketHeight<self.sprocketHeight and self.sprocketHeight<self.maxSprocketHeight:
                print(f"Valid sprocket size {self.sprocketHeight}")
                locateHoleResult = 0 #Sprocket size within range
            elif self.sprocketHeight>self.maxSprocketHeight:
                print(f"Invalid sprocket size too big {self.sprocketHeight} max {self.maxSprocketHeight}")
                locateHoleResult = 2 #Sprocket size too big
                print("Setting result to 2 - sprocket size too big")
            else:
                print(f"probably not enough peaks found {len(self.peaks)}")
                self.sprocketHeight   = 0
                locateHoleResult = 1
        rX = self.findSprocketRight(x1, x2, self.sprocketHeight, cY)
        oldcX = self.cX
        oldcY = self.cY
        oldrX = self.rX
        if not rX:
            locateHoleResult = 3 #Right edge not detected
            print("Setting result to 3 - right edge not detected")
            #raise Exception(f"Setting result to 3 - can't find right edge {self.imagePathName}")
            return locateHoleResult
        else:

            self.cX = int(lX+rX/2)
            self.cY = cY
            self.rX = rX

        #locateHoleResult = 0
        #print(f"InnerLow {innerLow} InnerHigh {innerHigh} cY {cY} rX {rX}")
        #print(f"Found sprocket edge {locatedX} at {rX}")
        print("cY=", self.cY, "oldcY=", oldcY, "locateHoleResult=", locateHoleResult)
        p1 = (int(x2-x1), 0) 
        p2 = (int(x2-x1), self.dy-1) 
        print(f"Vertical line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (255, 0, 0), 3) #View X1 Blue
        p1 = (0, int(self.cY))
        p2 = (int(self.rX-x1), int(self.cY))
        print(f"Horizontal line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Horiz
        p1 = (int(self.rX-x1), 0) 
        p2 = (int(self.rX-x1), self.dy) 
        print(f"Vertical line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Vert
        #cv2.line(self.image, p1, p2, (0, 0, 255), 3) #Vert
        # show target midy
        p1 = (0, int(midy)) 
        p2 = (int(self.rX-x1), int(midy))
        print(f"MidY line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 0), 3)  # black line
        p1 = (0, int(midy+3))
        p2 = (int(self.rX-x1), int(midy+3))
        cv2.line(self.imageHoleCrop, p1, p2, (255, 255, 255), 3) # white line
        
        #self.imageHoleCrop = cv2.resize(self.imageHoleCrop, (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)


        cv2.imwrite(os.path.expanduser("~/ratiosprocketStrip.png"), self.imageHoleCrop)
        #cv2.imwrite(self.resultImagePath, self.imageHoleCrop)
        cv2.imwrite(os.path.expanduser("~/ratioimage.png"), self.image)
        #if locatedX:
        #    cv2.imwrite(os.path.expanduser("~/horizontalStrip.png"), horizontalStrip)  
        #self.locateHoleResult = locateHoleResult
        return locateHoleResult


    def locateSprocketHoleRatioOld(self):
        # Based on https://github.com/cpixip/sprocket_detection
        print(f"locateSprocketHole {self.image.shape} {self.imagePathName}")
        filterSize = 25                 # smoothing kernel - leave it untouched
        midy = self.midy
        self.peaks = []
        self.smoothedHisto = []
        #Find the left edge of sprocket
        #x1,x2 = self.findXRange()
        lX = self.findSprocketLeft()
        x1 = int(lX+(0.01*self.dx))#buffer for rough edge
        if not x1:
            cv2.imwrite(os.path.expanduser("~/ratioa.png"), self.image)
            cv2.imwrite(os.path.expanduser("~/ratiothresh.png"), self.thresh)
            locateHoleResult = 5 #can't find left edge
            print(f"Setting result to 5 - can't find left edge {self.imagePathName}")
            raise Exception(f"Setting result to 5 - can't find left edge {self.imagePathName}")
        else:
            x2 = x1+int(self.stdSprocketWidth*0.8)
            self.imageHoleCrop = self.image[:,int(x1):int(x1+2*(x2-x1)),:]
            self.imageHoleCropHide = self.image[:,int(x1):int(x2),:]

            #Create mask for white - may need to adjust
            hsvMargin=50
            image = cv2.cvtColor(self.imageHoleCropHide, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 255-hsvMargin], dtype="uint8")
            upper = np.array([255, hsvMargin, 255], dtype="uint8")
            mask = cv2.inRange(image, lower, upper)
            masked = cv2.bitwise_and(self.imageHoleCropHide, self.imageHoleCropHide, mask=mask)
            cv2.imwrite(os.path.expanduser("~/whitemsk.png"), mask)
            cv2.imwrite(os.path.expanduser("~/masked.png"), masked)


            cv2.imwrite(os.path.expanduser("~/imageHoleCropHide.png"), self.imageHoleCropHide)

            #sprocketEdges = np.absolute(cv2.Sobel(self.imageHoleCropHide,cv2.CV_64F,0,1,ksize=3))
            sprocketEdges = np.absolute(cv2.Sobel(mask,cv2.CV_64F,0,1,ksize=3))

            #histogram     = np.mean(sprocketEdges,axis=(1,2))
            histogram     = np.mean(sprocketEdges,axis=(1))
            
            self.smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)
        Frame.ratioX1 = x1
        Frame.ratioX2 = x2
        #Find the y search range
        #sprocketStart, sprocketSize = self.findSprocket(x1,x2)
        #raise Exception(f"Stopping here")
        y1, y2 = self.findYRange(x1,x2)
        if not y1 and y2:
            locateHoleResult = 4 #Cant find sprocket/gap pattern
            print("Setting result to 4 - can't match pattern")
        #Locate sprocket in reduced range
        else:
            print(f"Searching for sprocket x1 {x1} x2 {x2} y1 {y1} y2 {y2}")
            cY, self.sprocketHeight = self.findSprocketSize(y1, y2)
            if self.minSprocketHeight<self.sprocketHeight and self.sprocketHeight<self.maxSprocketHeight:
                print(f"Valid sprocket size {self.sprocketHeight}")
                locateHoleResult = 0 #Sprocket size within range
            elif self.sprocketHeight>self.maxSprocketHeight:
                print(f"Invalid sprocket size too big {self.sprocketHeight}")
                locateHoleResult = 2 #Sprocket size too big
                print("Setting result to 2 - sprocket size too big")
            else:
                print(f"probably not enough peaks found {len(self.peaks)}")
                self.sprocketHeight   = 0
                locateHoleResult = 1
        rX = self.findSprocketRight(x1, x2, self.sprocketHeight, cY)
        oldcX = self.cX
        oldcY = self.cY
        oldrX = self.rX
        if not rX:
            locateHoleResult = 3 #Right edge not detected
            print("Setting result to 3 - right edge not detected")
            #raise Exception(f"Setting result to 3 - can't find right edge {self.imagePathName}")
            return locateHoleResult
        else:

            self.cX = int(lX+rX/2)
            self.cY = cY
            self.rX = rX

        #locateHoleResult = 0
        #print(f"InnerLow {innerLow} InnerHigh {innerHigh} cY {cY} rX {rX}")
        #print(f"Found sprocket edge {locatedX} at {rX}")
        print("cY=", self.cY, "oldcY=", oldcY, "locateHoleResult=", locateHoleResult)
        p1 = (int(x2-x1), 0) 
        p2 = (int(x2-x1), self.dy-1) 
        print(f"Vertical line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (255, 0, 0), 3) #View X1 Blue
        p1 = (0, int(self.cY))
        p2 = (int(self.rX-x1), int(self.cY))
        print(f"Horizontal line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Horiz
        p1 = (int(self.rX-x1), int(y1)) 
        p2 = (int(self.rX-x1), int(y2)) 
        print(f"Vertical line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Vert
        #cv2.line(self.image, p1, p2, (0, 0, 255), 3) #Vert
        # show target midy
        p1 = (0, int(midy)) 
        p2 = (int(self.rX-x1), int(midy))
        print(f"MidY line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 0), 3)  # black line
        p1 = (0, int(midy+3))
        p2 = (int(self.rX-x1), int(midy+3))
        cv2.line(self.imageHoleCrop, p1, p2, (255, 255, 255), 3) # white line
        
        #self.imageHoleCrop = cv2.resize(self.imageHoleCrop, (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)


        cv2.imwrite(os.path.expanduser("~/ratiosprocketStrip.png"), self.imageHoleCrop)
        #cv2.imwrite(self.resultImagePath, self.imageHoleCrop)
        cv2.imwrite(os.path.expanduser("~/ratioimage.png"), self.image)
        #if locatedX:
        #    cv2.imwrite(os.path.expanduser("~/horizontalStrip.png"), horizontalStrip)  
        #self.locateHoleResult = locateHoleResult
        return locateHoleResult




#=========================================================================
    def findSprocketLeft(self):
        returnlX = self.lX
        filmEdge = self.filmEdge
        searchRange = self.stdSprocketWidth*1.2#450 #may need to adjust with image size
        ratioThresh = 0.1 #may need to adjust with film format
        filmEdgeTrigger = 0.01
        #searchStart = int(self.holeCrop.x1-searchRange)
        searchStart = int(0.12*self.dx)#int(0.05*self.dx) #to the right of left edge of films
        searchStart =220 #TODO set frame edge should only be full black on film edge
        searchEnd = int(searchStart+searchRange)
        hMin = 0
        sMin = 0
        vMin = 54
        hMax = 179
        sMax = 85
        vMax = 255
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.threshrg = cv2.inRange(hsv, lower, upper) #TODO why am I using a different thresh??

        hsvMargin=50
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 255-hsvMargin], dtype="uint8")
        upper = np.array([255, hsvMargin, 255], dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        #self.threshrgmsk = cv2.bitwise_and(self.image, self.image, mask=mask)
        self.threshrgmsk = mask

        _, self.thresh = cv2.threshold(self.image, Frame.whiteThreshold, 255, 0)
        cv2.imwrite(os.path.expanduser("~/convertfail.png"),self.thresh)
        cv2.imwrite(os.path.expanduser("~/convertfailrg.png"),self.threshrg)
        cv2.imwrite(os.path.expanduser("~/convertfailrgmsk.png"),self.threshrgmsk)
        hsv = None
        step = 1
        print(f"Finding left edge from {searchStart} to {searchEnd}")
        for x1 in range(searchStart,searchEnd,step):
            #strip = self.thresh[:,int(x1):int(x1+step),] #TODO reduce y size
            #print(self.thresh.shape)
            #print(self.threshrg.shape)
            
            #strip = self.thresh[:,int(x1):int(x1+step),] #TODO reduce y size
            strip = self.threshrgmsk[:,int(x1):int(x1+step),] #TODO reduce y size
            
            ratio = float(np.sum(strip == 255)/(self.dy*step))
            print(f"x {x1} ratio {ratio} {np.sum(strip == 255)} dx {self.dx*step}")
            p1 = (int(x1), int(0)) 
            p2 = (int(x1), int(self.dy)) 
            cv2.line(self.image, p1, p2, (128, 128, 128), 1) #grey to film edge
            if ratio<filmEdgeTrigger:
                print(f"Found edge at {x1} with ratio {ratio}")
                filmEdge=x1
                self.filmEdge = filmEdge
                p1 = (int(x1), int(0)) 
                p2 = (int(x1), int(self.dy)) 
                cv2.line(self.image, p1, p2, (255, 0, 0), 2) #Vert
                break
        step = int(5*self.ScaleFactor) #increase for faster
        searchStart = int(filmEdge+(0.015*self.dx))#buffer for rough edge
        searchEnd = int(searchStart+searchRange)
        for x1 in range(searchStart,searchEnd,step):
            strip = self.threshrgmsk[:,int(x1):int(x1+step),]
            ratio = float(np.sum(strip == 255)/(self.dy*step))
            print(f"Ratio calc result {ratio} white {np.sum(strip==255)} / steparea {self.dy*step}")
            #print(f"x {x1} ratio {ratio} {np.sum(strip == 255)} dx {self.dy*step}")
            p1 = (int(x1), int(0)) 
            p2 = (int(x1), int(self.dy)) 
            cv2.line(self.image, p1, p2, (255, 255, 255), 3) #Vert
            if ratio>ratioThresh and filmEdge:
                cv2.imwrite(os.path.expanduser("~/testx.png"), self.image)
                print(f"Final lX {x1} ratio {ratio}")
                returnlX = x1+(step/2)
                self.lX = returnlX
                break
        #if not filmEdge:
        #    cv2.imwrite(os.path.expanduser("~/leftedgefail.png"), self.image)
        return returnlX

    def findSprocketRight(self, x1, x2, sprocketHeight, cY):
        if sprocketHeight==0:
            return 0
        rx1 = x1+ 0.1*(x2-x1) #skip left edge shadow
        rx2 = x1 + 1.5*self.stdSprocketWidth
        #rx2 = x1 + 2*self.self.stdSprocketWidth
        ry = int(0.8*sprocketHeight)
        ry1 = cY-ry//2
        ry2 = cY+ry//2
        print(f"strip dimensions format {self.format} ry1 {ry1} ry2 {ry2} rx1 {rx1} rx2 {rx2} - cY {cY} shape {self.image.shape}")
        horizontalStrip = self.image[int(ry1):int(ry2),int(rx1):int(rx2),:]
        horizontalEdges = np.absolute(cv2.Sobel(horizontalStrip,cv2.CV_64F,1,0,ksize=3))
        histoHori       = np.mean(horizontalEdges,axis=(0,2))
        smoothedHori    = cv2.GaussianBlur(histoHori,(1,5),0)
        maxPeakValueH   = smoothedHori.max()
        thresholdHori   = self.innerThresh*maxPeakValueH
        cv2.imwrite(os.path.expanduser("~/hori.png"), horizontalStrip)
        cv2.imwrite(os.path.expanduser("~/imgcheck.png"), self.image)
        #cv2.imwrite(os.path.expanduser("~/horicheck.png"), self.image[int(ry1):int(ry2),int(rx1):int(rx2),:])#self.image[0:1858,456:848,:])
        plt.plot(smoothedHori)
        plt.axhline(thresholdHori, color='cyan', linewidth=1)
        plt.savefig(os.path.expanduser("~/horihist.png"))
        plt.clf()
        triggered = False
        for x in range(int((x2-x1)//2),len(smoothedHori)):
            if smoothedHori[x]<thresholdHori:
                triggered = True
            if smoothedHori[x]>thresholdHori and triggered:         
                rX = x+rx1             
                return rX
        return 0 

    def locateSprocketHoleThresh(self):
        print(f"{self.imagePathName}")
        calcAreaSize = (self.stdSprocketHeight)*(self.stdSprocketWidth)
        area_size=int(0.8*calcAreaSize)#60000
        localImg = self.image.copy()
        #cv2.imwrite(os.path.expanduser("~/contoursb4cut.png"),img)
        x1 = int(self.findSprocketLeft() - (15*self.ScaleFactor))
        x2 = int(x1 + (self.stdSprocketWidth) + (40*self.ScaleFactor))
        self.threshX1 = x1
        self.threshX2 = x2
        self.threshY1 = 0
        self.threshY2 = self.dy
        Frame.ratioX1 = x1
        Frame.ratioX2 = x2
        print(f"Thresh checking boundaries x1 {self.threshX1} x2 {self.threshX2} y1 {self.threshY1} y2 {self.threshY2}")
        img = localImg[self.threshY1:self.threshY2, self.threshX1:self.threshX2]
        #img = self.image[self.threshY1:self.threshY2, self.threshX1:self.threshX2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #self.whiteTreshold = self.getWhiteThreshold()
        #self.whiteThreshold = self.getWhiteThreshold(self.threshImg)
        #self.whiteThreshold=220
        print(f"Checking white threshold before use {Frame.whiteThreshold}")
        print(f"area size {area_size}")
        ret, self.imageHoleCrop = cv2.threshold(img, Frame.whiteThreshold, 255, 0) 
        contours, hierarchy = cv2.findContours(self.imageHoleCrop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lenContours = len(contours)
        locateHoleResult = 1 
        oldcX = self.cX
        oldcY = self.cY
        oldrX = self.rX
        self.area = area_size
        #minDist = self.dy
        print(f"Calc interhole dist dy {self.dy} {self.stdSprocketHeight} {self.ratio}")
        print(f"intehole dist = {self.stdSprocketHeight*self.ratio}")
        minDist = self.stdSprocketHeight*self.ratio #estimate interhold dist
        for l in range(lenContours):
            cnt = contours[l]
            area = cv2.contourArea(cnt)
            #print(f"{l} {area}")
            if area > 0.1*area_size:
                x,y,w,h = cv2.boundingRect(cnt)
                #print(f"y {x} y {y} w {w} h {h}")
                print(f"{l} {area} {cv2.boundingRect(cnt)} {self.midy} - {y+0.5*h} = {abs(self.midy-(y+0.5*h))} mindist {minDist}")
            if area > area_size:
                x,y,w,h = cv2.boundingRect(cnt)
                print(f"y {x} y {y} w {w} h {h}")
                print(f"{l} {area} {cv2.boundingRect(cnt)} {self.midy} - {y+0.5*h} = {abs(self.midy-(y+0.5*h))}")
                dist = abs(self.midy-(y+0.5*h))
                if area > 3*area_size:#TODO too high - should return code and jump to another method
                    locateHoleResult = 2 # very large contour found == no film
                    #raise Exception(f"very large contour found == no film {area} > 3*{area_size}")
                elif dist<minDist:
                    print(f"found better at {dist}")
                    locateHoleResult = 0 # hole found
                    self.area = area
                    bestCont = cnt
                    minDist=dist
        if locateHoleResult == 0:      
            M = cv2.moments(bestCont)
            print(f"calculating cY from {M['m01'] / M['m00']} + {self.threshY1} = {int(M['m01'] / M['m00'])+self.threshY1} should be like {y+0.5*h}")
            try:
                self.cX = int(M["m10"] / M["m00"])+self.threshX1
                self.cY = int(M["m01"] / M["m00"])+self.threshY1
                x,y,w,h = cv2.boundingRect(bestCont)
                self.rX = x+w+self.threshX1
                print(f"calculating cY from {M['m01'] / M['m00']} + {self.threshY1} = {int(M['m01'] / M['m00'])+self.threshY1} should be like {y+0.5*h}")
            
                #holeDist = 225
                #if cY > holeDist : # distance between holes
                #    print("cY=", cY)
                #    locateHoleResult = 4 # 2. hole found
                #    cY = cY - holeDist
                resultImage = cv2.cvtColor(self.imageHoleCrop, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(resultImage, cnt, -1, (0,255,0), 3)
                cv2.imwrite(os.path.expanduser("~/contours.png"),resultImage)
                #break
            except ZeroDivisionError:
                print("no center")
                locateHoleResult = 3 # no center
                self.cX = oldcX
                self.cY = oldcY # midy
                self.rX = oldrX
        else :
            resultImage = cv2.cvtColor(self.imageHoleCrop, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(resultImage, contours, -1, (0,255,0), 3)
            cv2.imwrite(os.path.expanduser("~/contours_fail.png"),resultImage)
            self.cX = oldcX
            self.cY = oldcY  
                  
        #print("cY=", self.cY, "oldcY=", oldcY, "locateHoleResult=", locateHoleResult)
        print(f"result {locateHoleResult} cX {self.cX} cY {self.cY} rX {self.rX}")
        p1 = (0, int(self.cY)) 
        p2 = (int(self.threshX2-self.threshX1), int(self.cY))
        # print(p1, p2)
        cv2.line(resultImage, p1, p2, (0, 0 ,255), 3)
        #p1 = (int(self.cX-self.threshX1), 0) 
        #p2 = (int(self.cX-self.threshX1), self.holeCrop.y2-self.holeCrop.y1) 
        #print(p1, p2)
        #cv2.line(resultImage, p1, p2, (0, 255, 0), 3)
        p1 = (int(self.rX-self.threshX1), 0) 
        p2 = (int(self.rX-self.threshX1), self.threshY2-self.threshY1) 
        #print(p1, p2)
        cv2.line(resultImage, p1, p2, (0, 0, 255), 3)
        # show target midy
        p1 = (0, int(self.midy)) 
        p2 = (int(self.threshX2-self.threshX1), int(self.midy))
        cv2.line(resultImage, p1, p2, (0, 0, 0), 3)  # black line
        p1 = (0, int(self.midy+3))
        p2 = (int(self.threshX2-self.threshX1), int(self.midy+3))
        cv2.line(resultImage, p1, p2, (255, 255, 255), 3) # white line
        cv2.imwrite(os.path.expanduser("~/thresh_output.png"),resultImage)
        #cv2.imwrite(self.resultImagePath, resultImage)
        self.imageHoleCrop = resultImage
        #self.locateHoleResult = locateHoleResult
        return locateHoleResult
    

#==========================================================================

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

    def cropAll(self, progress) :
        self.restartCrop = False
        frameNo = 0
        os.chdir(Film.scanFolder)
        fileList = sorted(glob.glob('*.jpg'))
        self.scanFileCount = len(fileList)
        for fn in fileList:
            outFileName = os.path.join(Film.cropFolder, f"frame{frameNo:06}.jpg")
            if not os.path.isfile(outFileName):
                frame = Frame(os.path.join(Film.scanFolder, fn))
                frame.cropPic()
                cv2.imwrite(outFileName, frame.imageCropped)
                self.curFrameNo = frameNo
                if progress is not None:
                    if progress(frame) == 0:
                        break
            frameNo = frameNo+1
                    
    def makeFilm(self, filmName, progressReport, filmDone) :
        self.progressReport = progressReport
        self.filmDone = filmDone
        os.chdir(Film.cropFolder)
        filmPathName = os.path.join(Film.filmFolder, filmName) + '.avi'#'.mp4'
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


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
            Film.resolution = config[Ini.film]['resolution']
            Film.s8_framerate = config[Ini.film].getint('s8_framerate')
            Film.s8_stepsPrFrame = config[Ini.film].getint('s8_steps_pr_frame')
            Film.r8_framerate = config[Ini.film].getint('r8_framerate')
            Film.r8_stepsPrFrame = config[Ini.film].getint('r8_steps_pr_frame')
            
            Frame.format = Film.format
            Frame.outerThresh = config[Ini.frame].getFloat('outerThresh')
            Frame.innerThresh = config[Ini.frame].getFloat('innerThresh')
            Frame.s8_minSprocketSize = config[Ini.frame].getint('s8_minSprocketSize')
            Frame.s8_maxSprocketSize = config[Ini.frame].getint('s8_maxSprocketSize')
            Frame.s8_midx = config[Ini.frame].getint('s8_midx')
            Frame.s8_midy = config[Ini.frame].getint('s8_midy')
            Frame.r8_minSprocketSize = config[Ini.frame].getint('r8_minSprocketSize')
            Frame.r8_maxSprocketSize = config[Ini.frame].getint('r8_maxSprocketSize')
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
        config[Ini.frame]['s8_minSprocketSize'] = str(Frame.s8_minSprocketSize)
        config[Ini.frame]['s8_maxSprocketSize'] = str(Frame.s8_maxSprocketSize)
        config[Ini.frame]['s8_midx'] = str(Frame.s8_midx)
        config[Ini.frame]['s8_midy'] = str(Frame.s8_midy)
        config[Ini.frame]['r8_minSprocketSize'] = str(Frame.r8_minSprocketSize)
        config[Ini.frame]['r8_maxSprocketSize'] = str(Frame.r8_maxSprocketSize)
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
    ViewHeight = 1464#1232

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
    s8_frameCrop = Rect("frame_crop", 2, -107, 2+1453, 1040-107)
    s8_holeCrop = Rect("hole_crop", 75, 0, 110, 2463) 
    s8_minSprocketSize = 40
    s8_maxSprocketSize = 58
    s8_midx = 64
    s8_midy = 240
    r8_frameCrop = Rect("frame_crop", 146, 28, 146+814, 28+565)
    r8_holeCrop = Rect("hole_crop", 90, 0, 240, 276)
    r8_minSprocketSize = 40
    r8_maxSprocketSize = 58
    r8_midx = 64
    r8_midy = 120
    ScaleFactor = 1.0 # overwritten by initScaleFactor()
    outerThresh = 0.65
    innerThresh = 0.3
    s8_template = [60,155]
    r8_template = [60,155]
    def initScaleFactor():
        Frame.ScaleFactor = Camera.ViewWidth/640.0 
          
    def __init__(self, imagePathName=None,*,image=None):
        self.imagePathName = imagePathName
        if image is None and imagePathName is not None :
            self.imagePathName = imagePathName
            self.image = cv2.imread(imagePathName)
        elif image is not None :
            self.image = image
        self.dy,self.dx,_ = self.image.shape
        self.ScaleFactor = self.dx/640.0
        print(f"Scalefactor {Frame.ScaleFactor}")
        if Frame.format == "s8":
            self.minSprocketSize = Frame.s8_minSprocketSize*self.ScaleFactor
            self.maxSprocketSize = Frame.s8_maxSprocketSize*self.ScaleFactor
            self.holeCrop = Rect("hole_crop", Frame.s8_holeCrop.x1*self.ScaleFactor, 0, Frame.s8_holeCrop.x2*self.ScaleFactor, self.dy-1)
            self.template = Frame.s8_template
        else:
            self.minSprocketSize = Frame.r8_minSprocketSize*self.ScaleFactor
            self.maxSprocketSize = Frame.r8_maxSprocketSize*self.ScaleFactor
            self.holeCrop = Rect("hole_crop", Frame.r8_holeCrop.x1*self.ScaleFactor, 0, Frame.r8_holeCrop.x2*self.ScaleFactor, self.dy-1)
            self.template = Frame.r8_template
        self.midx = 115*self.ScaleFactor   # always overwitten 
        self.midy = self.dx//2#240*self.ScaleFactor 
        self.cX = self.midx 
        self.cY = self.midy    
        
        self.locateHoleResult = 1
        print(f"init complete {self.__dict__}")
        
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
        return self.convert_cv_qt(self.imageCropped, dest)
        
    def getHoleCrop(self) :
        return self.convert_cv_qt(self.imageHoleCrop)

    def calcCrop(self):
        #self.locateHoleResult = self.locateSprocketHoleNew(Frame.holeMinArea)
        self.locateHoleResult = self.locateSprocketHoleNew()
        print(f"Calc crop self.cY {self.cY} + Frame.holeCrop.y1 {Frame.holeCrop.y1} = self.cY + Frame.holeCrop.y1 {self.cY + Frame.holeCrop.y1}")
        print(f"Frame.ScaleFactor {Frame.ScaleFactor}")
        print(f"Frame.frameCrop.y1 {Frame.frameCrop.y1}")
        print(f"int((self.cY + Frame.holeCrop.y1) * Frame.ScaleFactor)+Frame.frameCrop.y1 {int((self.cY + Frame.holeCrop.y1) * Frame.ScaleFactor)+Frame.frameCrop.y1}")
        #x = int((self.cX + Frame.holeCrop.x1) * Frame.ScaleFactor)+Frame.frameCrop.x1
        #y = int((self.cY + Frame.holeCrop.y1) * Frame.ScaleFactor)+Frame.frameCrop.y1 
        x = int(self.cX + (Frame.frameCrop.x1 * Frame.ScaleFactor))
        y = int(self.cY + (Frame.frameCrop.y1 * Frame.ScaleFactor)) 
        self.p1 = (x, y)
        self.p2 = (x+Frame.frameCrop.getXSize(), y+Frame.frameCrop.getYSize())
        print(f"crop self.p1 {self.p1} self.p2 {self.p2}")
        
    def getCropOutline(self, dest=None):
        self.calcCrop()
        cv2.rectangle(self.image, self.p1, self.p2, (0, 255, 0), 10)
        wp1 = (round(Frame.whiteCrop.x1 * Frame.ScaleFactor), round(Frame.whiteCrop.y1 * Frame.ScaleFactor))
        wp2 = (round(Frame.whiteCrop.x2 * Frame.ScaleFactor), round(Frame.whiteCrop.y2 * Frame.ScaleFactor))
        cv2.rectangle(self.image, wp1, wp2, (60, 240, 240), 10)
        return self.convert_cv_qt(self.image, dest)
        
    def cropPic(self):
        self.calcCrop()
        self.imageCropped = self.image[self.p1[1]:self.p2[1], self.p1[0]:self.p2[0]]
     
    def getWhiteThreshold(self, imageSmall):
        img = imageSmall[Frame.whiteCrop.y1:Frame.whiteCrop.y2, Frame.whiteCrop.x1:Frame.whiteCrop.x2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        planes = cv2.split(img)
        histSize = 256 #  [Establish the number of bins]
        histRange = (0, 256) # Set the range
        hist = cv2.calcHist(planes, [0], None, [histSize], histRange, accumulate=False)    
        okPct = (Frame.whiteCrop.y2-Frame.whiteCrop.y1)*(Frame.whiteCrop.x2-Frame.whiteCrop.x1)/100.0*5
        wco = 220 # Default value - will work in most cases
        for i in range(128,256) :
            if hist[i] > okPct :
                wco = i-8 #6
                break
        return wco        

    # Based on https://github.com/cpixip/sprocket_detection
    # initial testing had issues with selecting point between 2 holes TODO
    # Added initial range check
    # Return values:
    # 0: hole found, 1: hole not found, 2: hole to large, 3: no center
    def locateSprocketHoleNew(self):
        print(f"locateSprocketHoleNew {self.image.shape}")
        thresholds = [0.5,0.3]          # edge thresholds; first one higher, second one lower
        filterSize = 25                 # smoothing kernel - leave it untouched
        #print(f"Check Frame.holeCrop.y1 {Frame.holeCrop.y1} Frame.holeCrop.y2 {Frame.holeCrop.y2}")
        midy = self.midy
        dy = self.dy
        y1 = self.holeCrop.y1
        y2 = self.holeCrop.y2
        template = self.template
        #x1 = int(Frame.holeCrop.x1*Frame.ScaleFactor)
        #x2 = int(Frame.holeCrop.x2*Frame.ScaleFactor)
        x1 = Frame.holeCrop.x1
        x2 = Frame.holeCrop.x2
        print(f"x1 {x1} x2 {x2}")
        print(f"y1 {y1} y2 {y2}")
        self.imageHoleCrop = self.image[:,x1:x1+2*(x2-x1),:]
        self.imageHoleCropHide = self.image[:,x1:x2,:]
        sprocketEdges = np.absolute(cv2.Sobel(self.imageHoleCropHide,cv2.CV_64F,0,1,ksize=3))
        histogram     = np.mean(sprocketEdges,axis=(1,2))
        smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)
        maxPeakValue   = smoothedHisto[y1:y2].max()
        outerThreshold = Frame.outerThresh*maxPeakValue
        innerThreshold = Frame.innerThresh*maxPeakValue
        outerLow       = y1
        peaks = []
        gap_thresh = 10*Frame.ScaleFactor #How close to template values at scaled
        for y in range(y1,y2):
            if smoothedHisto[y]<outerThreshold and smoothedHisto[y+1]>outerThreshold:
                peaks.append(y)
        print(f"Peaks {peaks} midy {midy}")
        #Find a range containing a sprocket closest to centre by comparing sprocket/not sprocket gaps with template
        #Try forward first. Backwards may be necessary if less peaks are detected
        #detected sprocket must be within 0.3 of the frame
        frameLocated=False
        for i in range(0,len(peaks)-2):
            #try forwards
            print(f"{peaks[i+1]-peaks[i]} - {template[0]*Frame.ScaleFactor} = {peaks[i+1]-peaks[i]-(template[0]**Frame.ScaleFactor):.2f}")
            print(f"and {peaks[i+2]-peaks[i+1]} -{template[1]} = {peaks[i+2]-peaks[i+1]-template[1]} ")
            if(abs(peaks[i+1]-peaks[i]-(template[0]*Frame.ScaleFactor))<gap_thresh) and (abs(peaks[i+2]-peaks[i+1]-(template[1]*Frame.ScaleFactor))<gap_thresh):
                print(f"Found hole starting at {peaks[i]} which is {(midy-peaks[i])/dy:.2f} from the centre")
                print(f"Check dist from centre {abs((midy-peaks[i])/dy):.2f}")
                if abs((midy-peaks[i])/dy)<0.3:
                    y1=int(peaks[i]-(template[0]*Frame.ScaleFactor))
                    y2=int(peaks[i]+(template[1]*Frame.ScaleFactor))
                    print(f"New y1 {y1} new y2 {y2}")
                    frameLocated = True
                    break
        if not frameLocated:
            for i in range(0,len(peaks)-2):
                #try backwards
                print(f"{peaks[i+1]-peaks[i]} - {template[1]*Frame.ScaleFactor} = {peaks[i+1]-peaks[i]-(template[1]**Frame.ScaleFactor):.2f}")
                print(f"and {peaks[i+2]-peaks[i+1]} -{template[0]} = {peaks[i+2]-peaks[i+1]-template[0]} ")
                if(abs(peaks[i+1]-peaks[i]-(template[1]*Frame.ScaleFactor))<gap_thresh) and (abs(peaks[i+2]-peaks[i+1]-(template[0]*Frame.ScaleFactor))<gap_thresh):
                    print(f"Found hole backwards starting at {peaks[i+1]} which is {(midy-peaks[i+1])/dy:.2f} from the centre")
                    print(f"Check dist from centre {abs((midy-peaks[i+1])/dy):.2f}")
                    if abs((midy-peaks[i+1])/dy)<0.3:
                        y1=int(peaks[i+1]-(template[0]*Frame.ScaleFactor))
                        y2=int(peaks[i+1]+(template[1]*Frame.ScaleFactor))
                        print(f"New y1 {y1} new y2 {y2}")
                        break
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
            searchCenter = int(outerLow + (0.5*self.minSprocketSize)) #give priority to top frame. Try to find internal location - probably can change to outerhigh-outerlow
            print(f"Between 2 frames, using top. Searching from {searchCenter}")
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
        #minSprocketSize = int(minSize)
        print(f"minSprocketSize {self.minSprocketSize}<sprocketSize {sprocketSize} {self.minSprocketSize<sprocketSize}")
        print(f"maxSprocketSize {self.maxSprocketSize}>sprocketSize {sprocketSize} {self.maxSprocketSize>sprocketSize}")
        print(f"and sprocketSize<(outerHigh-outerLow) {sprocketSize<(outerHigh-outerLow)}")
        if self.minSprocketSize<sprocketSize and sprocketSize<(outerHigh-outerLow) and sprocketSize<self.maxSprocketSize:
            cY = (innerHigh+innerLow)//2
            print(f"Valid sprocket size {sprocketSize}")
            locateHoleResult = 0
        elif sprocketSize>self.maxSprocketSize:
            cY = (innerHigh+innerLow)//2
            print(f"Invalid sprocket size too big {sprocketSize}")
            locateHoleResult = 2
        else:
            print(f"Why am I here with sprocketSize {sprocketSize}")
            cY = dy//2
            sprocketSize   = 0
            locateHoleResult = 1
        cX = 0
        locatedX = False
        if sprocketSize>0:
            rx1 = x1
            rx2 = x1 + 2*(x2-x1)
            ry = int(0.8*sprocketSize)
            ry1 = cY-ry//2
            ry2 = cY+ry//2
            horizontalStrip = self.image[ry1:ry2,rx1:rx2,:]
            horizontalEdges = np.absolute(cv2.Sobel(horizontalStrip,cv2.CV_64F,1,0,ksize=3))
            histoHori       = np.mean(horizontalEdges,axis=(0,2))
            smoothedHori    = cv2.GaussianBlur(histoHori,(1,5),0)
            maxPeakValueH   = smoothedHori.max()
            thresholdHori   = thresholds[1]*maxPeakValueH
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

        print("cY=", self.cY, "oldcY=", oldcY, "locateHoleResult=", locateHoleResult)
 
        p1 = (0, int(self.cY))
        p2 = (int(x2-x1), int(self.cY))
        # print(p1, p2)
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Horiz
        p1 = (int(self.cX), int(y1)) 
        p2 = (int(self.cX), int(y2)) 
        # print(p1, p2)
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Vert
        #cv2.line(self.image, p1, p2, (0, 0, 255), 3) #Vert
        # show target midy
        p1 = (0, int(midy)) 
        p2 = (int(x2-x1), int(midy))
        cv2.line(self.imageHoleCrop, p1, p2, (0, 255, 0), 1)  # black line
        p1 = (0, int(midy+1))
        p2 = (int(x2-x1), int(midy+1))
        cv2.line(self.imageHoleCrop, p1, p2, (255, 255, 255), 1) # white line
        self.imageHoleCrop = cv2.resize(self.imageHoleCrop, (0,0), fx=1/Frame.ScaleFactor, fy=1/Frame.ScaleFactor)

        plt.plot(smoothedHisto)
        plt.axvline(cY, color='blue', linewidth=1)
        plt.axvline(searchCenter, color='orange', linewidth=1)
        plt.axvline(innerHigh, color='green', linewidth=1)
        plt.axvline(innerLow, color='red', linewidth=1)
        plt.axvline(outerHigh, color='purple', linewidth=1)
        plt.axvline(outerLow, color='gray', linewidth=1)
        plt.axhline(innerThreshold, color='cyan', linewidth=1)
        plt.axhline(outerThreshold, color='olive', linewidth=1)
        plt.xlim([0, dy])
        #plt.show()
        plt.savefig("/home/warwickh/my_cv2hist.png")
        plt.xlim(y1,y2)
        #plt.show()
        plt.savefig("/home/warwickh/my_cv2hist_lim.png")
        plt.clf()
        cv2.imwrite(f"/home/warwickh/sprocketStrip.png", self.imageHoleCrop)
        cv2.imwrite(f"/home/warwickh/image.png", self.image)
        if locatedX:
            cv2.imwrite(f"/home/warwickh/horizontalStrip.png", horizontalStrip)
           
        self.locateHoleResult = locateHoleResult
        return locateHoleResult
            
class Film:
    format = "s8"
    resolution = "720x540"
    s8_framerate = 24
    r8_framerate = 12
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
        frameNo = 0
        os.chdir(Film.scanFolder)
        fileList = sorted(glob.glob('*.jpg'))
        self.scanFileCount = len(fileList)
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
                "-framerate", str(Film.Framerate), 
                "-f", "image2",
                "-pattern_type", "sequence",
                "-i", os.path.join(Film.cropFolder, "frame%06d.jpg"),
                "-s", Film.Resolution,
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


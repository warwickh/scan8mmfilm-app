#from FilmScanModule import Rect
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import json

class sprocketHole:
    """provide multiple method of finding sprocket hole boundaries threshold, sobel edge and canny edge"""
    def __init__(self, 
                 frame
                 ):
        self.frame = frame
        print(f"Sprockethole init {self.frame.imagePathName}")
        self.image = self.frame.image.copy()
        self.dy,self.dx,_ = self.image.shape
        self.ScaleFactor = self.dx/640.0
        self.sprocketWidth = self.frame.sprocketWidth
        self.stdSprocketHeight = self.frame.stdSprocketHeight
        self.minSprocketHeight = self.stdSprocketHeight*self.dy*0.9
        self.maxSprocketHeight = self.stdSprocketHeight*self.dy*1.1
        self.format = self.frame.format
        self.ratio = self.frame.ratio
        self.locateHoleResult = 6
        self.cX = self.frame.cX
        self.cY = self.frame.cY
        self.rX = self.cX + self.sprocketWidth
        self.midy = self.frame.midy
        #self.whiteThreshold = 225


        #Thresh needs to see all of the hole         
        if self.format=="s8":
            self.threshX1 = 250
            self.threshX2 = 550
        else:
            self.threshX1 = 400
            self.threshX2 = 1050


        #self.threshImg = os.path.expanduser("~/whitethresh.png")
        if self.frame.imagePathName:
            self.imagePathName = self.frame.imagePathName
            self.threshImg = os.path.join(os.path.dirname(self.imagePathName),"whitethresh.png")
        else:
            self.imagePathName = None
            self.threshImg = "whitethresh.png"
        self.whiteThreshold = self.getWhiteThreshold(self.threshImg)
        self.resultImagePath = self.frame.resultImagePath
        #self.dy,self.dx,_ = self.image.shape
        #self.ScaleFactor = self.dx/640.0

    def process(self):
        self.results = {}
        methods = ["thresh","ratio","mask"]#,"canny"] #prioritise by speed
        for method in methods:
            self.image = self.frame.image.copy()
            start_time = time.time()
            if method=="thresh":
                result = self.locateSprocketHoleThresh()
                #Use threshold value to find sprockethole contour - should be one step and fast
            elif method=="ratio":
                result = self.locateSprocketHoleRatio()
                #Find left edge using white profile (thresh value?)
                #Find Y range
                #Find sprocket hole
                #Find right edge
            elif method=="mask":
                result = self.locateSprocketHoleMask()
                #Same as ratio but using white masked value
            elif method=="canny":
                result = self.locateSprocketHoleCanny()
                #Same as mask but add canny edge detection
            end_time = time.time()
            self.locateHoleResult = result
            self.image = None
            elapsed_time = end_time - start_time
            #self.results[method]={}
            #self.results[method]['result'] = result
            #self.results[method]['time'] = elapsed_time
            if result==0:
            #    self.results[method]['cY'] = self.cY
                #results[method]['cX'] = self.cX
            #    self.results[method]['rX'] = self.rX
                return result, self.cY, self.rX
            print(f"{method} took {elapsed_time:.04f} with result {self.locateHoleResult}")
            #if self.locateHoleResult==0:
            #    print(f"Got a result {self.locateHoleResult}")
                #break
        #self.results = {self.imagePathName:results}
        #print(self.results)
        #with open( os.path.expanduser("~/frames.json"), "a") as outfile:
        #    outfile.write(json.dumps(self.results, indent=4))

    def getWhiteThreshold(self, threshFilename):
        #threshFilename = os.path.expanduser("~/thresh_test.png")
        print(f"Checking for threshold file at {threshFilename} {os.path.exists(threshFilename)}")
        if not os.path.exists(threshFilename):
            #testImg = cv2.resize(self.image.copy(), (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)
            #testImg = cv2.resize(self.image, (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)
            testImg = self.image.copy()
            #cv2.imwrite("d:\\howzitlook.png", testImg)
            #testImg = cv2.resize(testImg, (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)
            cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("Resized_Window", 800, 600)
            roi=cv2.selectROI("Resized_Window", testImg)
            cropped = testImg[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            cv2.imwrite(threshFilename, cropped)
            cv2.destroyAllWindows()
        img = cv2.imread(threshFilename)
        dy,dx,_ = img.shape
        #img = imageSmall[self.frame.whiteCrop.y1:self.frame.whiteCrop.y2, self.frame.whiteCrop.x1:self.frame.whiteCrop.x2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        planes = cv2.split(img)
        histSize = 256 #  [Establish the number of bins]
        histRange = (0, 256) # Set the range
        hist = cv2.calcHist(planes, [0], None, [histSize], histRange, accumulate=False) 
        okPct = dy*dx/100.0*5
        #okPct = (self.frame.whiteCrop.y2-self.frame.whiteCrop.y1)*(self.frame.whiteCrop.x2-self.frame.whiteCrop.x1)/100.0*5
        wco = 220 # Default value - will work in most cases
        for i in range(128,256) :
            if hist[i] > okPct :
                wco = i-8 #6
                break
        print(f"Found threshold {wco} from {threshFilename}")
        return wco   

    def locateSprocketHoleThresh(self):
        calcAreaSize = (self.stdSprocketHeight*self.dy)*(self.sprocketWidth*self.dx)
        area_size=0.8*calcAreaSize#60000
        #rint(f"Calc area for {self.format} {(self.stdSprocketHeight*self.dy)*(self.sprocketWidth*self.dx):.02f}")
        #self.imageSmall = cv2.resize(self.image, (640, 480))
        # the image crop with the sprocket hole 
        localImg = self.image.copy()
        #cv2.imwrite(os.path.expanduser("~/contoursb4cut.png"),img)

        x1 = int(self.findSprocketLeft() - (20*self.ScaleFactor))
        x2 = int(x1 + (self.sprocketWidth*self.dx) + (40*self.ScaleFactor))
        self.threshX1 = x1
        self.threshX2 = x2
        print(f"Thresh checking boundaries x1 {self.threshX1} x2 {self.threshX2} y1 {self.frame.holeCrop.y1} y2 {self.frame.holeCrop.y2}")
        img = localImg[self.frame.holeCrop.y1:self.frame.holeCrop.y2, self.threshX1:self.threshX2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #self.whiteTreshold = self.getWhiteThreshold()
        #self.whiteThreshold = self.getWhiteThreshold(self.threshImg)
        print(f"Checking white threshold before use {self.whiteThreshold}")
        ret, self.imageHoleCrop = cv2.threshold(img, self.whiteThreshold, 255, 0) 
        contours, hierarchy = cv2.findContours(self.imageHoleCrop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lenContours = len(contours)
        locateHoleResult = 1 
        oldcX = self.cX
        oldcY = self.cY
        oldrX = self.rX
        self.area = area_size
        minDist = self.dy
        for l in range(lenContours):
            cnt = contours[l]
            area = cv2.contourArea(cnt)
            if area > area_size:
                x,y,w,h = cv2.boundingRect(cnt)
                print(f"{l} {area} {cv2.boundingRect(cnt)} {y} {self.midy} {abs(self.midy-(y+0.5*h))}")
                dist = abs(self.midy-(y+0.5*h))
                if area > 3*area_size:
                    locateHoleResult = 2 # very large contour found == no film
                elif dist<minDist:
                    print(f"found better at {dist}")
                    locateHoleResult = 0 # hole found
                    self.area = area
                    bestCont = cnt
                    minDist=dist
        if locateHoleResult == 0:      
            M = cv2.moments(bestCont)
            # print(M)
            try:
                self.cX = int(M["m10"] / M["m00"])+self.threshX1
                self.cY = int(M["m01"] / M["m00"])+self.frame.holeCrop.y1
                x,y,w,h = cv2.boundingRect(bestCont)
                self.rX = x+w+self.threshX1
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
        p1 = (0, self.cY) 
        p2 = (self.threshX2-self.threshX1, self.cY)
        # print(p1, p2)
        cv2.line(resultImage, p1, p2, (0, 0 ,255), 3)
        #p1 = (int(self.cX-self.threshX1), 0) 
        #p2 = (int(self.cX-self.threshX1), self.frame.holeCrop.y2-self.frame.holeCrop.y1) 
        #print(p1, p2)
        #cv2.line(resultImage, p1, p2, (0, 255, 0), 3)
        p1 = (int(self.rX-self.threshX1), 0) 
        p2 = (int(self.rX-self.threshX1), self.frame.holeCrop.y2-self.frame.holeCrop.y1) 
        #print(p1, p2)
        cv2.line(resultImage, p1, p2, (0, 0, 255), 3)
        # show target midy
        p1 = (0, self.midy) 
        p2 = (self.threshX2-self.threshX1, self.midy)
        cv2.line(resultImage, p1, p2, (0, 0, 0), 1)  # black line
        p1 = (0, self.midy+1)
        p2 = (self.threshX2-self.threshX1, self.midy+1)
        cv2.line(resultImage, p1, p2, (255, 255, 255), 1) # white line
        cv2.imwrite(os.path.expanduser("~/thresh_output.png"),resultImage)
        cv2.imwrite(self.resultImagePath, resultImage)
        
        self.locateHoleResult = locateHoleResult
        return locateHoleResult
    
    #==========================================================================================================
    
    def findSprocketLeft(self):
        returnX1 = 0
        returnX2 = 0
        searchRange = 450 #may need to adjust with image size
        ratioThresh = 0.1 #may need to adjust with film format
        #searchStart = int(self.holeCrop.x1-searchRange)
        searchStart = int(0.1*self.dx)#int(0.05*self.dx) #to the right of left edge of films
        searchStart =450
        searchEnd = int(searchStart+searchRange)
        step = int(10*self.ScaleFactor)
        countSteps = 0 #check that we're not taking too long
        hMin = 0
        sMin = 0
        vMin = 54
        hMax = 179
        sMax = 85
        vMax = 255
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.thresh = cv2.inRange(hsv, lower, upper)
        hsv = None
        for x1 in range(searchStart,searchEnd,step):
            strip = self.thresh[:,int(x1):int(x1+step),]
            ratio = float(np.sum(strip == 255)/(self.dx*step))
            print(f"x {x1} ratio {ratio} {np.sum(strip == 255)} dx {self.dx*step}")
            p1 = (int(x1), int(0)) 
            p2 = (int(x1), int(self.dy)) 
            cv2.line(self.image, p1, p2, (255, 255, 255), 3) #Vert
            countSteps+=1
            if ratio>ratioThresh:
                cv2.imwrite(os.path.expanduser("~/testx.png"), self.image)
                print(f"Final x {x1} ratio {ratio} steps {countSteps}")
                returnX1 = x1+(step/2)
                break
        return returnX1

    def findSprocketRight(self, x1, x2, sprocketHeight, cY):
        if sprocketHeight==0:
            return 0
        rx1 = x1
        rx2 = x1 + 1.5*(x2-x1)
        #rx2 = x1 + 2*self.sprocketWidth*self.dx
        ry = int(0.8*sprocketHeight)
        ry1 = cY-ry//2
        ry2 = cY+ry//2
        print(f"strip dimensions format {self.format} ry1 {ry1} ry2 {ry2} rx1 {rx1} rx2 {rx2} - cY {cY} shape {self.image.shape}")
        horizontalStrip = self.image[int(ry1):int(ry2),int(rx1):int(rx2),:]
        horizontalEdges = np.absolute(cv2.Sobel(horizontalStrip,cv2.CV_64F,1,0,ksize=3))
        histoHori       = np.mean(horizontalEdges,axis=(0,2))
        smoothedHori    = cv2.GaussianBlur(histoHori,(1,5),0)
        maxPeakValueH   = smoothedHori.max()
        thresholdHori   = self.frame.innerThresh*maxPeakValueH
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
                cX = x+x1             
                return cX
        return 0 

    def findSprocketLowRange(self):
        #Looks for low areas in the histogram - not very effective but may work well with canny histo instead
        y1=0
        y2=int(self.dy-1)
        if len(self.smoothedHisto)==0:
            return None
        peaks = []
        innerHigh = 0
        innerLow = 0
        outerHigh = y2
        outerLow = y1
        minPeakValue  = self.smoothedHisto[y1:y2].min()
        maxPeakValue  = self.smoothedHisto[y1:y2].max()
        valueRange = (maxPeakValue-minPeakValue)*0.02
        self.innerThreshold = self.frame.innerThresh*maxPeakValue
        print(f"Range size {valueRange:.02f} min {minPeakValue:.02f} max {minPeakValue+valueRange:.02f}")
        lowRanges = []
        for y in range(y1,y2):
            if self.smoothedHisto[y]<minPeakValue+valueRange and self.smoothedHisto[y]>=minPeakValue:
                peaks.append(y)
            else:
                if len(peaks)>100:
                    print(f"values caught {len(peaks)} mid {peaks[len(peaks)//2]}")
                    lowRanges.append(peaks)
                    peaks = []
            if self.smoothedHisto[y]==minPeakValue:
                self.trough=y
        print(f"Found {len(lowRanges)} ranges that could be sprocket holes")
        plt.plot(self.smoothedHisto)
        midPoint=0
        for r in lowRanges:
            plt.axvline(r[0], color='cyan', linewidth=1)
            plt.axvline(r[len(r)-1], color='cyan', linewidth=1)
            midPoint = r[len(r)//2]
            plt.axvline(midPoint, color='red', linewidth=1)
            print(f"Found potential sprocket at {midPoint} using low range which is {(self.midy-midPoint)/self.dy:.2f} from the centre {self.midy}")
            if abs((self.midy-midPoint))/self.dy<0.2:
                print(f"Using range of size {len(r)} and midPoint {midPoint}")
                break
        if not midPoint==0: #I have a valid range so find the limits of the sprocket hole
            innerLow = midPoint
            outerHigh = int(midPoint+(1.5*self.stdSprocketHeight*self.dy))
            outerLow = int(midPoint-(1.5*self.stdSprocketHeight*self.dy))
            print(f"searchCenter {midPoint} outerLow {outerLow} outerHigh {outerHigh} self.stdSprocketHeight*self.dy {self.stdSprocketHeight*self.dy} self.stdSprocketHeight {self.stdSprocketHeight}")
            for y in range(midPoint,outerLow,-1):
                if self.smoothedHisto[y]>self.innerThreshold:
                    innerLow = y
                    break
            innerHigh = midPoint
            for y in range(midPoint,outerHigh):
                if self.smoothedHisto[y]>self.innerThreshold:
                    innerHigh = y
                    break
        plt.axvline((innerHigh+innerLow)//2, color='blue', linewidth=1)
        plt.axvline(midPoint, color='orange', linewidth=1)
        plt.axvline(innerHigh, color='purple', linewidth=1)
        plt.axvline(innerLow, color='purple', linewidth=1)
        plt.axvline(outerHigh, color='green', linewidth=1)
        plt.axvline(outerLow, color='green', linewidth=1)
        plt.axhline(self.innerThreshold, color='cyan', linewidth=1)
        plt.axvline(self.trough, color='pink', linewidth=1)
        plt.xlim([0, self.dy])
        plt.savefig(os.path.expanduser("~/my_cv2hist.png"))
        plt.xlim(outerLow,outerHigh)
        plt.savefig(self.frame.hist_path)#os.path.expanduser("~/my_cv2hist_lim.png"))
        plt.clf()
        return innerHigh, innerLow

    def locateSprocketHoleLow(self):
        # Based on https://github.com/cpixip/sprocket_detection
        print(f"locateSprocketHoleLow {self.image.shape} {self.imagePathName}")
        filterSize = 25                 # smoothing kernel - leave it untouched
        midy = self.midy
        self.peaks = []
        self.smoothedHisto = []
        x1 = self.findSprocketLeft()
        if not x1:
            #Failed to find left edge - save some debug info
            cv2.imwrite(os.path.expanduser("~/a.png"), self.image)
            cv2.imwrite(os.path.expanduser("~/thresh.png"), self.thresh)
            locateHoleResult = 5 #can't find left edge
            print(f"Setting result to 5 - can't find left edge {self.imagePathName}")
            #raise Exception(f"Setting result to 5 - can't find left edge {self.imagePathName}")
            return locateHoleResult
        else:
            x2 = x1+int(self.sprocketWidth*self.dx)#*0.8
            self.imageHoleCrop = self.image[:,int(x1):int(x1+1.5*(x2-x1)),:] #bigger so we can see
            self.imageHoleCropHide = self.image[:,int(x1):int(x2),:] #For processing only
            #self.canny = self.imageHoleCropCanny[:,int(x1+(0.01*self.dx)):int(x1+(0.03*self.dx)),:] #For testing
            sprocketEdges = np.absolute(cv2.Sobel(self.imageHoleCropHide,cv2.CV_64F,0,1,ksize=3))
            histogram     = np.mean(sprocketEdges,axis=(1,2))
            self.smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)
        print(f"New search for sprocket x1")
        innerHigh, innerLow = self.findSprocketLowRange()
        self.sprocketHeight = innerHigh-innerLow
        cY = (innerHigh+innerLow)//2
        sprocketHeightTol = 0.04
        print(f"Using sprocket {self.sprocketHeight} in range {(self.stdSprocketHeight-sprocketHeightTol)*self.dy:.2f} to {(self.stdSprocketHeight+sprocketHeightTol)*self.dy:.2f}")
        if self.sprocketHeight==0:
            print(f"Invalid sprocket size zero {self.sprocketHeight}")
            locateHoleResult = 2
            cv2.imwrite(os.path.expanduser("~/2.png"), self.image)
            #raise Exception(f"Setting result to 2 - can't find valid sprocket range for {self.imagePathName}")
            return locateHoleResult
        elif self.sprocketHeight>(self.stdSprocketHeight-sprocketHeightTol)*self.dy and self.sprocketHeight<(self.stdSprocketHeight+sprocketHeightTol)*self.dy:
            print(f"Valid sprocket size {self.sprocketHeight}")
            locateHoleResult = 0 #Sprocket size within range
        else:
            print(f"Invalid sprocket size {self.sprocketHeight}")
            self.sprocketHeight = 0
            locateHoleResult = 1
            cv2.imwrite(os.path.expanduser("~/lowimage.png"), self.image) 
            cv2.imwrite(os.path.expanduser("~/lowthresh.png"), self.thresh)
            #raise Exception(f"Setting result to 1 - sprocket out of range for {self.imagePathName}")
            return locateHoleResult
        cX = self.findSprocketRight(x1, x2, self.sprocketHeight, cY)
        #oldcX = self.cX
        oldcY = self.cY
        if not cX:
            locateHoleResult = 3 #Right edge not detected
            print("Setting result to 3 - right edge not detected")
            #raise Exception(f"Setting result to 3 - can't find right edge for {self.imagePathName}")
            return locateHoleResult
        else:
            self.cX = cX
            self.cY = cY

        #Markup crosshairs
        print("cY=", self.cY, "oldcY=", oldcY, "locateHoleResult=", locateHoleResult)
        p1 = (int(x2-x1), 0) 
        p2 = (int(x2-x1), self.dy-1) 
        #print(f"Vertical line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (255, 0, 0), 3) #View X1 Blue
        p1 = (0, int(self.cY))
        p2 = (int(self.cX-x1), int(self.cY))
        #print(f"Horizontal line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Horiz
        p1 = (int(self.cX-x1), int(innerLow)) 
        p2 = (int(self.cX-x1), int(innerHigh)) 
        #print(f"Vertical line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 255), 3) #Vert
        p1 = (0, int(midy)) 
        p2 = (int(self.cX-x1), int(midy))
        #print(f"MidY line points {p1} {p2}")
        cv2.line(self.imageHoleCrop, p1, p2, (0, 0, 0), 3)  # black line
        p1 = (0, int(midy+3))
        p2 = (int(self.cX-x1), int(midy+3))
        cv2.line(self.imageHoleCrop, p1, p2, (255, 255, 255), 3) # white line
        self.imageHoleCrop = cv2.resize(self.imageHoleCrop, (0,0), fx=1/self.ScaleFactor, fy=1/self.ScaleFactor)
        cv2.imwrite(os.path.expanduser("~/lowsprocketStrip.png"), self.imageHoleCrop)
        cv2.imwrite(os.path.expanduser("~/lowimage.png"), self.image) 
        cv2.imwrite(os.path.expanduser("~/lowthresh.png"), self.thresh)
        self.locateHoleResult = locateHoleResult
        return locateHoleResult

    def findYRangeNew(self, x1, x2):
        y1=0
        y2=self.dy-1
        returnY1 = None
        returnY2 = None
        if len(self.smoothedHisto)==0:
            return None, None
        maxPeakValue   = self.smoothedHisto[y1:y2].max()
        minPeakValue   = self.smoothedHisto[y1:y2].min()

        fullrange = maxPeakValue-minPeakValue
        finalZ = 0
        plt.plot(self.smoothedHisto)
        for z in range(int(maxPeakValue),int(maxPeakValue-(fullrange*0.5)),int(0-fullrange*0.1)):
            print(f"Trying {z} from {maxPeakValue} to {maxPeakValue-(fullrange*0.5)}")
            peaks = []
            trough = None
            for y in range(y1,y2):
                if self.smoothedHisto[y]<z and self.smoothedHisto[y+1]>z:
                    peaks.append(y)
                if self.smoothedHisto[y]==minPeakValue:
                    self.trough=y
            print(f"Peaks at {z:.2f} {peaks} thresh {self.outerThreshold:.2f}")
            if len(peaks)>2 and len(peaks)<6 and finalZ==0:
                plt.axhline(z, color='blue', linewidth=1)
                self.peaks=peaks
                print(f"Got enough peaks {self.peaks} {len(self.peaks)} at {z:.2f} midy {self.midy} thresh {self.outerThreshold:.2f} trough at {self.trough}")
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
                    if sprocketStart and abs((self.midy-sprocketStart)/self.dy)<0.25:
                        returnY1=int(sprocketStart-(1*self.maxSprocketHeight))
                        returnY2=min(self.dy,int(sprocketStart+(1.5*self.maxSprocketHeight)))
                        print(f" Sprocket within range, so new y1 {returnY1} new y2 {returnY2}")
                        finalZ = z
                        break
        plt.savefig(os.path.expanduser("~/maskYsearch.png"))
        plt.clf()
        return returnY1, returnY2
        """
        self.outerThreshold = self.frame.outerThresh*maxPeakValue
        self.innerThreshold = self.frame.innerThresh*maxPeakValue
        plt.plot(self.smoothedHisto)
        #thresh_vals = [outerThreshold+10, outerThreshold+5, outerThreshold, outerThreshold-5, outerThreshold-10]
        #thresh_vals = [self.outerThreshold, self.outerThreshold-5, self.outerThreshold-10,self.outerThreshold+5]
        thresh_vals = [self.outerThreshold, self.outerThreshold*0.9, self.outerThreshold*0.8,self.outerThreshold*0.7, self.outerThreshold*0.6,self.outerThreshold*0.5]
        print(f"Thresh vals {thresh_vals}")
        finalZ = 0
        for z in thresh_vals:
            plt.axhline(z, color='blue', linewidth=1)
            peaks = []
            trough = None
            for y in range(y1,y2):
                if self.smoothedHisto[y]<z and self.smoothedHisto[y+1]>z:
                    peaks.append(y)
                if self.smoothedHisto[y]==minPeakValue:
                    self.trough=y
            print(f"Peaks at {z:.2f} {peaks} thresh {self.outerThreshold:.2f}")
            if len(peaks)>2 and len(peaks)<6 and finalZ==0:
                self.peaks=peaks
                finalZ = z
                print(f"Got enough peaks {self.peaks} {len(self.peaks)} at {z:.2f} midy {self.midy} thresh {self.outerThreshold:.2f} trough at {self.trough}")
                break
            for i in self.peaks:
                plt.axvline(i, color='blue', linewidth=1)
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
        return returnY1, returnY2
        """



#=====================================================================================================================
    #Working implementation with histogram using ratios
    def findYRange(self, x1, x2):
        y1=0
        y2=self.dy-1
        returnY1 = None
        returnY2 = None
        if len(self.smoothedHisto)==0:
            return None, None
        maxPeakValue   = self.smoothedHisto[y1:y2].max()
        minPeakValue   = self.smoothedHisto[y1:y2].min()

        fullrange = maxPeakValue-minPeakValue

        self.outerThreshold = self.frame.outerThresh*maxPeakValue
        self.innerThreshold = self.frame.innerThresh*maxPeakValue
        plt.plot(self.smoothedHisto)
        #thresh_vals = [outerThreshold+10, outerThreshold+5, outerThreshold, outerThreshold-5, outerThreshold-10]
        #thresh_vals = [self.outerThreshold, self.outerThreshold-5, self.outerThreshold-10,self.outerThreshold+5]
        thresh_vals = [self.outerThreshold, self.outerThreshold*0.9, self.outerThreshold*0.8,self.outerThreshold*0.7, self.outerThreshold*0.6,self.outerThreshold*0.5]
        print(f"Thresh vals {thresh_vals}")
        finalZ = 0
        for z in thresh_vals:
            plt.axhline(z, color='blue', linewidth=1)
            peaks = []
            trough = None
            for y in range(y1,y2):
                if self.smoothedHisto[y]<z and self.smoothedHisto[y+1]>z:
                    peaks.append(y)
                if self.smoothedHisto[y]==minPeakValue:
                    self.trough=y
            print(f"Peaks at {z:.2f} {peaks} thresh {self.outerThreshold:.2f}")
            if len(peaks)>2 and len(peaks)<6 and finalZ==0:
                self.peaks=peaks
                finalZ = z
                print(f"Got enough peaks {self.peaks} {len(self.peaks)} at {z:.2f} midy {self.midy} thresh {self.outerThreshold:.2f} trough at {self.trough}")
                break
            for i in self.peaks:
                plt.axvline(i, color='blue', linewidth=1)
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
        return returnY1, returnY2

    def findSprocketSize(self, y1, y2):
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
        plt.savefig(self.frame.hist_path)#os.path.expanduser("~/my_cv2hist_lim.png"))
        #self.histogram = cv2.imread(os.path.expanduser("~/my_cv2hist_lim.png"))
        plt.clf()
        return cY, sprocketHeight

    def locateSprocketHoleRatio(self):
        # Based on https://github.com/cpixip/sprocket_detection
        print(f"locateSprocketHole {self.image.shape} {self.imagePathName}")
        filterSize = 25                 # smoothing kernel - leave it untouched
        midy = self.midy
        self.peaks = []
        self.smoothedHisto = []
        #Find the left edge of sprocket
        #x1,x2 = self.findXRange()
        x1 = self.findSprocketLeft()
        if not x1:
            cv2.imwrite(os.path.expanduser("~/ratioa.png"), self.image)
            cv2.imwrite(os.path.expanduser("~/ratiothresh.png"), self.thresh)
            locateHoleResult = 5 #can't find left edge
            print(f"Setting result to 5 - can't find left edge {self.imagePathName}")
            raise Exception(f"Setting result to 5 - can't find left edge {self.imagePathName}")
        else:
            x2 = x1+int(self.sprocketWidth*self.dx)#*0.8
            self.imageHoleCrop = self.image[:,int(x1):int(x1+2*(x2-x1)),:]
            self.imageHoleCropHide = self.image[:,int(x1):int(x2),:]
            sprocketEdges = np.absolute(cv2.Sobel(self.imageHoleCropHide,cv2.CV_64F,0,1,ksize=3))
            histogram     = np.mean(sprocketEdges,axis=(1,2))
            self.smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)
        #Find the y search range
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
        cX = self.findSprocketRight(x1, x2, self.sprocketHeight, cY)
        oldcX = self.cX
        oldcY = self.cY
        if not cX:
            locateHoleResult = 3 #Right edge not detected
            print("Setting result to 3 - right edge not detected")
            #raise Exception(f"Setting result to 3 - can't find right edge {self.imagePathName}")
            return locateHoleResult
        else:

            self.cX = cX
            self.cY = cY

        #locateHoleResult = 0
        #print(f"InnerLow {innerLow} InnerHigh {innerHigh} cY {cY} cX {cX}")
        #print(f"Found sprocket edge {locatedX} at {cX}")
        print("cY=", self.cY, "oldcY=", oldcY, "locateHoleResult=", locateHoleResult)
        p1 = (int(x2-x1), 0) 
        p2 = (int(x2-x1), self.dy-1) 
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


        cv2.imwrite(os.path.expanduser("~/ratiosprocketStrip.png"), self.imageHoleCrop)
        cv2.imwrite(self.resultImagePath, self.imageHoleCrop)
        cv2.imwrite(os.path.expanduser("~/ratioimage.png"), self.image)
        #if locatedX:
        #    cv2.imwrite(os.path.expanduser("~/horizontalStrip.png"), horizontalStrip)  
        self.locateHoleResult = locateHoleResult
        return locateHoleResult
    
    #================================================================================================================
    #Ratio with masking
    def locateSprocketHoleMask(self):
        # Based on https://github.com/cpixip/sprocket_detection
        print(f"locateSprocketHoleMask {self.image.shape} {self.imagePathName}")
        filterSize = 25                 # smoothing kernel - leave it untouched
        midy = self.midy
        self.peaks = []
        self.smoothedHisto = []
        #Find the left edge of sprocket
        #x1,x2 = self.findXRange()
        x1 = self.findSprocketLeft()
        if not x1:
            cv2.imwrite(os.path.expanduser("~/ratioa.png"), self.image)
            cv2.imwrite(os.path.expanduser("~/ratiothresh.png"), self.thresh)
            locateHoleResult = 5 #can't find left edge
            print(f"Setting result to 5 - can't find left edge {self.imagePathName}")
            raise Exception(f"Setting result to 5 - can't find left edge {self.imagePathName}")
        else:
            x2 = x1+int(self.sprocketWidth*self.dx)#*0.8
            self.imageHoleCrop = self.image[:,int(x1):int(x1+2*(x2-x1)),:]
            
            self.imageHoleCropHide = self.image[:,int(x1):int(x2),:]
            sprocketEdges = np.absolute(cv2.Sobel(self.imageHoleCropHide,cv2.CV_64F,0,1,ksize=3))
            histogram     = np.mean(sprocketEdges,axis=(1,2))
            self.smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)          
            

            tUpper=200
            tLower=100
            hsvMargin=50
            filterSize=25
            self.imageHoleCropHide = self.image[:,int(x1+(0.01*self.dx)):int(x1+(0.03*self.dx)),:] #For testing
            #Create mask for white - may need to adjust
            image = cv2.cvtColor(self.imageHoleCropHide, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 255-hsvMargin], dtype="uint8")
            upper = np.array([255, hsvMargin, 255], dtype="uint8")
            mask = cv2.inRange(image, lower, upper)
            masked = cv2.bitwise_and(self.imageHoleCropHide, self.imageHoleCropHide, mask=mask)
            cv2.imwrite(os.path.expanduser("~/whitemsk.png"), mask)
            cv2.imwrite(os.path.expanduser("~/masked.png"), masked)

            sprocketEdges = np.absolute(cv2.Sobel(mask,cv2.CV_64F,0,1,ksize=3))
            histogram     = np.mean(sprocketEdges,axis=(1))
            self.smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)
            #plt.plot(maskSmoothedHisto)
            #plt.savefig(os.path.expanduser("~/maskSmoothedHistoSobel.png"))
            #plt.clf()
        #Find the y search range
        y1, y2 = self.findYRangeNew(x1,x2)
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
        cX = self.findSprocketRight(x1, x2, self.sprocketHeight, cY)
        oldcX = self.cX
        oldcY = self.cY
        if not cX:
            locateHoleResult = 3 #Right edge not detected
            print("Setting result to 3 - right edge not detected")
            #raise Exception(f"Setting result to 3 - can't find right edge {self.imagePathName}")
            return locateHoleResult
        else:

            self.cX = cX
            self.cY = cY

        #locateHoleResult = 0
        #print(f"InnerLow {innerLow} InnerHigh {innerHigh} cY {cY} cX {cX}")
        #print(f"Found sprocket edge {locatedX} at {cX}")
        print("cY=", self.cY, "oldcY=", oldcY, "locateHoleResult=", locateHoleResult)
        p1 = (int(x2-x1), 0) 
        p2 = (int(x2-x1), self.dy-1) 
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


        cv2.imwrite(os.path.expanduser("~/ratiosprocketStrip.png"), self.imageHoleCrop)
        cv2.imwrite(self.resultImagePath, self.imageHoleCrop)
        cv2.imwrite(os.path.expanduser("~/ratioimage.png"), self.image)
        #if locatedX:
        #    cv2.imwrite(os.path.expanduser("~/horizontalStrip.png"), horizontalStrip)  
        self.locateHoleResult = locateHoleResult
        return locateHoleResult

    #================================================================================================================
    #Test an implementation using canny to refine the histo

    def locateSprocketHoleCanny(self):
        tUpper=200
        tLower=100
        hsvMargin=50
        filterSize=25
        x1 = self.findSprocketLeft()
        if not x1:
            cv2.imwrite(os.path.expanduser("~/acanny.png"), self.image)
            cv2.imwrite(os.path.expanduser("~/threshcanny.png"), self.thresh)
            locateHoleResult = 5 #can't find left edge
            print(f"Setting result to 5 - can't find left edge {self.imagePathName}")
            #raise Exception(f"Setting result to 5 - can't find left edge {self.imagePathName}")
            return locateHoleResult
        else:
            #x2 = x1+int(self.sprocketWidth*self.dx)#*0.8
            #self.imageHoleCrop = self.image[:,int(x1):int(x1+2*(x2-x1)),:]
            #self.imageHoleCropHide = self.image[:,int(x1):int(x2),:]
            #sprocketEdges = np.absolute(cv2.Sobel(self.imageHoleCropHide,cv2.CV_64F,0,1,ksize=3))
            #histogram     = np.mean(sprocketEdges,axis=(1,2))
            #self.smoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)

            img = self.image[:,int(x1+(0.01*self.dx)):int(x1+(0.03*self.dx)),:] #For testing
            #Create mask for white - may need to adjust
            image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 255-hsvMargin], dtype="uint8")
            upper = np.array([255, hsvMargin, 255], dtype="uint8")
            mask = cv2.inRange(image, lower, upper)
            masked = cv2.bitwise_and(img,img, mask= mask)
            cv2.imwrite(os.path.expanduser("~/whitemsk.png"), mask)
            cv2.imwrite(os.path.expanduser("~/masked.png"), masked)

            sprocketEdges = np.absolute(cv2.Sobel(mask,cv2.CV_64F,0,1,ksize=3))
            histogram     = np.mean(sprocketEdges,axis=(1))
            maskSmoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)
            plt.plot(maskSmoothedHisto)
            plt.savefig(os.path.expanduser("~/maskSmoothedHistoSobel.png"))
            plt.clf()

            sprocketEdges = np.absolute(cv2.Sobel(masked,cv2.CV_64F,0,1,ksize=3))
            histogram     = np.mean(sprocketEdges,axis=(1,2))
            maskSmoothedHisto = cv2.GaussianBlur(histogram,(1,filterSize),0)
            plt.plot(maskSmoothedHisto)
            plt.savefig(os.path.expanduser("~/maskedSmoothedHistoSobel.png"))
            plt.clf()

            imgMasked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            imgMasked = cv2.GaussianBlur(img, (5, 5), 0)
            cv2.imwrite(os.path.expanduser("~/imgMasked.png"), masked)
            edges = cv2.Canny(imgMasked,tLower,tUpper)
            cannyprocketEdges = np.absolute(cv2.Sobel(edges,cv2.CV_64F,0,1,ksize=3))
            cannyHistogram     = np.mean(cannyprocketEdges,axis=(1))
            cannySmoothedHisto = cv2.GaussianBlur(cannyHistogram,(1,filterSize),0)
            plt.plot(cannySmoothedHisto)
            plt.savefig(os.path.expanduser("~/my_masked.png"))
            plt.clf()
            cv2.imwrite(os.path.expanduser("~/cn_cv2out.png"), edges)
            return 6

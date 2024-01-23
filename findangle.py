import numpy as np 
from PIL import Image
from FilmScanModule import Ini, Camera, Frame, Film, getAdjustableRects

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import cv2
  
def combine_images(first_image, second_image):
    img1 = cv2.imread(first_image)
    img2 = cv2.imread(second_image)
    h_img = cv2.hconcat([img1, img2])
    return h_img

if __name__ == "__main__":
    Ini.loadConfig()
    #folder = "C:\\Users\\F98044d\\Downloads\\dup_test"
    #folder = os.path.expanduser("~/scanframes/crop/roll4a")
    folder = "D:\\roll4e"
    folder = "C:\\Users\\F98044d\\Videos\\git\\scan8mmfilm-app\\scanframes"
    os.chdir(folder)
    fileList = sorted(glob.glob('scan*.jpg'))
    lastImage = None
    with open(os.path.expanduser(f"~/angles.txt"),'w') as logFile:
        for fn in fileList:
            print(f"{fn}")
            #currentImage = os.path.join(folder,fn)
            #frame = Frame(os.path.join(folder,fn))
            #frame.format="s8"
            #locateHoleResult = frame.locateSprocketHole()
            #print(locateHoleResult)
            #frame.cropPic()
            #cropped = frame.imageCropped
            img = cv2.imread(os.path.join(folder,fn))#frame.image
            """
            x1 = -10
            x2 = 1750
            y1 = -130
            y2 = 1200

            x = int(frame.cX + (x1 * frame.ScaleFactor))
            y = int(frame.cY + (y1 * frame.ScaleFactor)) 
            p1 = (x, y)
            p2 = (x+(x2-x1), y+(y2-y1))
            print(f"crop self.p1 {p1} self.p2 {p2}")
            cropped = img[p1[1]:p2[1], p1[0]:p2[0]] 




            #cropped = img[y1:y2,x1:x2]
            outName = fn.split(".")[0]
            print(outName)
            print(cropped.shape)
            print(frame.holeCrop.__dict__)
            window_name = outName

            img = cv2.imread("messigray.jpg",0)
            """
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
            tlines = lsd.detect(thresh)[0] #Position 0 of the returned tuple are the detected lines
            
            drawn_img = lsd.drawSegments(img,lines)
            for line in lines:
                print(line)
            tdrawn_img = lsd.drawSegments(thresh,tlines)
            outName = fn.split(".")[0]
            cv2.imwrite(os.path.expanduser(f"~/angle/lines_{outName}.png"),drawn_img)
            cv2.imwrite(os.path.expanduser(f"~/angle/tlines_{outName}.png"),tdrawn_img)
            #cv2.imshow("LSD",drawn_img )
            #cv2.waitKey(0)
            """
            #cv2.imshow(window_name, cropped) 
            #cv2.waitKey(0)
            cv2.imwrite(os.path.expanduser(f"~/angle/{outName}.png"), cropped)
            img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            cv2.imwrite(os.path.expanduser(f"~/angle/th3.png"),th3)

            #thresh = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            #ret,thresh = cv2.threshold(thresh,127,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for l in range(len(contours)):
                cnt = contours[l]
                area = cv2.contourArea(cnt)
                if area > 30000:
                    cv2.drawContours(th3, contours[l], -1, (0,255,0), 3)
                    cv2.imwrite(os.path.expanduser(f"~/angle/contours_{l}.png"),th3)
            """
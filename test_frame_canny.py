from FilmScanModule import Ini, Camera, Frame, Film, getAdjustableRects
from CannyEdgeDetector import cannyEdgeDetector
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

if __name__ == "__main__":
    Ini.loadConfig()
    folder = os.path.expanduser("~/scanframes/inbound-test")
    os.chdir(folder)
    fileList = sorted(glob.glob('0009*.jpg'))
    for fn in fileList:
        print(f"{fn}")
        frame = Frame(os.path.join(folder,fn))
        locateHoleResult = frame.locateSprocketHole()
        img = frame.canny

        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([22, 93, 0], dtype="uint8")
        upper = np.array([45, 255, 255], dtype="uint8")
        margin = 50
        lower = np.array([0, 0, 255-margin], dtype="uint8")
        upper = np.array([255, margin, 255], dtype="uint8")
        
        mask = cv2.inRange(image, lower, upper)
        masked = cv2.bitwise_and(img,img, mask= mask)
        cv2.imwrite(os.path.expanduser("~/whitemsk.png"), mask)
        cv2.imwrite(os.path.expanduser("~/masked.png"), masked)

        #img = cv2.imread(os.path.expanduser("~/gray.png"))
        #img = mask
        img = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        #img = rgb2gray(masked)
        cv2.imwrite(os.path.expanduser("~/cannyin.png"), img)
        
        detector = cannyEdgeDetector([img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
        imgs_final = detector.detect()
        img_out = imgs_final[0]
        filterSize = 25                 # smoothing kernel - leave it untouched
        cannyprocketEdges = np.absolute(img_out)
        cannyHistogram     = np.mean(cannyprocketEdges,axis=(1))
        cannySmoothedHisto = cv2.GaussianBlur(cannyHistogram,(1,filterSize),0)
        plt.clf()
        plt.plot(cannySmoothedHisto)
        #plt.axvline(cY, color='blue', linewidth=1)
        #plt.axvline(searchCenter, color='orange', linewidth=1)
        #plt.axvline(innerHigh, color='green', linewidth=1)
        #plt.axvline(innerLow, color='red', linewidth=1)
        #plt.axvline(outerHigh, color='purple', linewidth=1)
        #plt.axvline(outerLow, color='gray', linewidth=1)
        #plt.axhline(innerThreshold, color='cyan', linewidth=1)
        #plt.axhline(outerThreshold, color='olive', linewidth=1)
        #plt.axvline(trough, color='pink', linewidth=1)
        #plt.xlim([0, dy])
        #plt.show()
        plt.savefig(os.path.expanduser("~/my_cannyhist_full.png"))
        plt.clf()
        cv2.imwrite(os.path.expanduser("~/cn_out.png"), img_out)
        #visualize(imgs_final, 'gray')

        edges = cv2.Canny(img,100,200)
        cannyprocketEdges = np.absolute(edges)
        cannyHistogram     = np.mean(cannyprocketEdges,axis=(1))
        cannySmoothedHisto = cv2.GaussianBlur(cannyHistogram,(1,filterSize),0)
        plt.clf()
        plt.plot(cannySmoothedHisto)
        #plt.axvline(cY, color='blue', linewidth=1)
        #plt.axvline(searchCenter, color='orange', linewidth=1)
        #plt.axvline(innerHigh, color='green', linewidth=1)
        #plt.axvline(innerLow, color='red', linewidth=1)
        #plt.axvline(outerHigh, color='purple', linewidth=1)
        #plt.axvline(outerLow, color='gray', linewidth=1)
        #plt.axhline(innerThreshold, color='cyan', linewidth=1)
        #plt.axhline(outerThreshold, color='olive', linewidth=1)
        #plt.axvline(trough, color='pink', linewidth=1)
        #plt.xlim([0, dy])
        #plt.show()
        plt.savefig(os.path.expanduser("~/my_cv2cannyhist_full.png"))
        plt.clf()
        cv2.imwrite(os.path.expanduser("~/cn_cv2out.png"), edges)
    #run_stab(fileList)

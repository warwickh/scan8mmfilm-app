from FilmScanModule import Ini,Frame
from CannyEdgeDetector import cannyEdgeDetector
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from Crop import crop

if __name__ == "__main__":
    Ini.loadConfig()
    folder = os.path.expanduser("~/scanframes/roll7")
    folder = "d:\\roll4e"#os.path.expanduser("~/scanframes/roll3a")
    os.chdir(folder)
    fileList = sorted(glob.glob('*.jpg'))
    #result_file = open( os.path.expanduser("~/frames.json"), "w")
    #result_file.close()
    for fn in fileList:
    #    print(f"{fn}")
        img = cv2.imread(os.path.join(folder,fn))
        roi=cv2.selectROI(img)
        print(roi)
        cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        cv2.imwrite("D:\\croptest\\out.png",cropped)
        
        #cropper = crop(os.path.join(folder,fn))
    #    frame = Frame(os.path.join(folder,fn))
    #    locateHoleResult = frame.locateSprocketHoleTest()
    #    print(f"Final result {locateHoleResult}")
    #    if not locateHoleResult==0:
    #        break

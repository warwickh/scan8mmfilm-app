from FilmScanModule import Ini,Frame
from CannyEdgeDetector import cannyEdgeDetector
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
#from Crop import crop

if __name__ == "__main__":
    Ini.loadConfig()
    folder = os.path.expanduser("~/scanframes/inbound-test")
    #folder = os.path.expanduser("~/scanframes/roll3a")
    os.chdir(folder)
    fileList = sorted(glob.glob('*0015.jpg'))
    result_file = open( os.path.expanduser("~/frames.json"), "w")
    result_file.close()
    for fn in fileList:
        print(f"{fn}")
        #cropper = crop(os.path.join(folder,fn))
        frame = Frame(os.path.join(folder,fn))
        locateHoleResult = frame.locateSprocketHoleTest()
        print(f"Final result {locateHoleResult}")

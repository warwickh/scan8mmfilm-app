from FilmScanModule import Ini, Camera, Frame, Film, getAdjustableRects
from CannyEdgeDetector import cannyEdgeDetector
import cv2
import glob
import os


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

if __name__ == "__main__":
    folder = os.path.expanduser("~/scanframes/bike")
    os.chdir(folder)
    fileList = sorted(glob.glob('*.jpg'))
    for fn in fileList:
        print(f"{fn}")
        frame = Frame(os.path.join(folder,fn))
        locateHoleResult = frame.locateSprocketHole()
        img = cv2.imread(os.path.expanduser("~/gray.png"))
        img = rgb2gray(img)
        detector = cannyEdgeDetector([img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
        imgs_final = detector.detect()
        cv2.imwrite(os.path.expanduser("~/cn_out.png"), imgs_final[0])
        #visualize(imgs_final, 'gray')

    #run_stab(fileList)

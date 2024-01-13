from FilmScanModule import Ini, Camera, Frame, Film, getAdjustableRects
import cv2
import glob
import os


if __name__ == "__main__":
    folder = "D:\\roll4f"
    os.chdir(folder)
    fileList = sorted(glob.glob('*.jpg'))
    for fn in fileList:
        frame = Frame(os.path.join(folder,fn))
        locateHoleResult = frame.locateSprocketHole()
    #run_stab(fileList)

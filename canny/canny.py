import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os

def callback(x):
    print(x)

img = cv2.imread(os.path.expanduser("~/gray.png"), 0) #read image as grayscale
ScaleFactor = 4
img = cv2.resize(img, (0,0), fx=1/ScaleFactor, fy=1/ScaleFactor)

canny = cv2.Canny(img, 255, 39) 

cv2.namedWindow('image') # make a window with name 'image'
cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

while(1):
    numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')

    canny = cv2.Canny(img, l, u)

cv2.destroyAllWindows()

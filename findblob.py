import cv2
import numpy as np
import os


def findX1(image):
    """find left side of sprocket holes"""
    _,dx,_ = image.shape
    
    searchRange = 450 #may need to adjust with image size
    ratioThresh = 0.05 #may need to adjust with film format
    
    #searchStart = int(self.holeCrop.x1-searchRange)
    searchStart = 0
    searchEnd = int(searchStart+searchRange)
    step = 10

    countSteps = 0 #check that we're not taking too long
    for x1 in range(searchStart,searchEnd,step):
        strip = image[:,int(x1):int(x1+step),:]
        #gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        #thresh = cv2.threshold(strip, 140, 255, cv2.THRESH_BINARY)[1]


        hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 212])
        upper_white = np.array([131, 255, 255])
        # Threshold the HSV image
        thresh = cv2.inRange(hsv, lower_white, upper_white)
        cv2.imwrite(os.path.join("C:\\Users\\F98044d\\Downloads\\dup_test_out",f"mask_{x1}.jpg"), thresh)

        ratio = float(np.sum(thresh == 255)/(dx*step))
        print(f"x {x1} ratio {ratio} {np.sum(thresh == 255)} dx {dx*step}")
        print(os.environ.get('HOMEPATH'))
        #cv2.imwrite(os.path.expanduser("~/testx.png"), thresh)
        p1 = (int(x1), int(0)) 
        p2 = (int(x1), int(dy)) 
        print(f"Vertical line points {p1} {p2}")
        cv2.line(image, p1, p2, (255, 255, 255), 3) #Vert
        cv2.imwrite(os.path.join("C:\\Users\\F98044d\\Downloads\\dup_test_out",f"strip_{x1}.jpg"), thresh)



        countSteps+=1
        if ratio>ratioThresh:
            cv2.imwrite(os.path.join("C:\\Users\\F98044d\\Downloads\\dup_test_out",f"final_strip_{x1}.jpg"), thresh)
            cv2.imwrite(os.path.join("C:\\Users\\F98044d\\Downloads\\dup_test_out",f"final_image_{x1}.jpg"), image)
            #cv2.imwrite(os.path.expanduser("~/testx.png"), thresh)
            print(f"x {x1} ratio {ratio} steps {countSteps}")
            return x1




folder = "C:\\Users\\F98044d\\Downloads\\dup_test"
folder = "C:\\Users\\F98044d\\Videos\\git\\scan8mmfilm-app"
for file in os.listdir(folder):
  if file=="scan000000.jpg":
    print(f"File {file}")
    imagePathName = os.path.join(folder,file)#"scan000000.jpg"
    image = cv2.imread(imagePathName)
    dy,dx,_ = image.shape
    x1 = findX1(image)

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #thresholding
    #thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1]
    """
    dy,dx,_ = image.shape
    step = 1
    ratioThresh = 0.15
    print(f"{100/dx}")
    print(f"{400/dx}")
    sprocketStartRange = [0.03,0.12]
    for x1 in range(int(sprocketStartRange[0]*dx),int(sprocketStartRange[1]*dx),step):
        #x1 = 80
        #x2 = x1+step
        #strip = image[:,int(x1):int(x1+step),:]
        gray = cv2.cvtColor(image[:,int(x1):int(x1+step),:], cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1]
        #number_of_white_pix = np.sum(thresh == 255)
        #number_of_black_pix = np.sum(thresh == 0) 
        ratio = float(np.sum(thresh == 255)/dx)
        print(f"x {x1} ratio {ratio}")
        if ratio>ratioThresh:
            cv2.imwrite(os.path.join("C:\\Users\\F98044d\\Downloads\\dup_test_out",f"strip_{x1}.jpg"), thresh)
            print(f"x {x1} ratio {ratio}")
            break
    # get indices of all white pixels
    whites = np.nonzero(thresh)
    for white in whites:
        print(white)
    # print the last white pixel in x-axis, 
    # which is obviously white
    value = thresh[whites[0][len(whites[0])-1]][whites[1][len(whites[1])-1]]


    p1 = (int(value), int(0)) 
    p2 = (int(value), int(dy)) 
    print(f"Vertical line points {p1} {p2}")
    cv2.line(thresh, p1, p2, (0, 0, 255), 3) #Vert
    cv2.imwrite(os.path.join("C:\\Users\\F98044d\\Downloads\\dup_test_out",file), thresh)
    """
import cv2
import numpy as np
import os

folder = "C:\\Users\\F98044d\\Downloads\\dup_test"
for file in os.listdir(folder):
  if file=="scan000000.jpg":
    print(f"File {file}")
    imagePathName = os.path.join(folder,file)#"scan000000.jpg"
    image = cv2.imread(imagePathName)
    dy,dx,_ = image.shape


    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #thresholding
    #thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1]

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
        number_of_white_pix = np.sum(thresh == 255)
        number_of_black_pix = np.sum(thresh == 0) 
        ratio = float(np.sum(thresh == 255)/dx)
        print(f"x {x1} ratio {ratio}")
        if ratio>ratioThresh:
            cv2.imwrite(os.path.join("C:\\Users\\F98044d\\Downloads\\dup_test_out",f"strip_{x1}.jpg"), thresh)
            print(f"x {x1} ratio {ratio}")
            break
    # get indices of all white pixels
    """
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
import cv2
import numpy as np
import os

class crop:
    def __init__(self, imageFilename):
        self.imageFilename = imageFilename
        self.img = cv2.resize(cv2.imread(imageFilename), (640,480))
        self.img_dup = np.copy(self.img)
        self.mouse_pressed = False
        #defining starting and ending point of rectangle (crop image region)
        self.starting_x=self.starting_y=self.ending_x=self.ending_y= -1
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.mousebutton)
        while True:
            cv2.imshow('image',self.img_dup)
            k = cv2.waitKey(1)
            if k==ord('c'):
                #remove these condition and try to play weird output will give you idea why its done
                if self.starting_y>self.ending_y:
                    self.starting_y,self.ending_y= self.ending_y,self.starting_y
                if self.starting_x>self.ending_x:
                    self.starting_x,self.ending_x= self.ending_x,self.starting_x
                if self.ending_y-self.starting_y>1  and self.ending_x-self.starting_x>0:
                    self.image = self.img[self.starting_y:self.ending_y,self.starting_x:self.ending_x]
                    cv2.imwrite(os.path.join(os.path.dirname(self.imageFilename),"whitethresh.png"), self.image)
                    self.img_dup= np.copy(self.image)
            elif k == ord('q'): 
                break
        cv2.destroyAllWindows()
        
    
    def mousebutton(self, event,x,y,flags,param):
            global img_dup , starting_x,starting_y,ending_x,ending_y,mouse_pressed
            #if left mouse button is pressed then takes the cursor position at starting_x and starting_y 
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mouse_pressed= True
                self.starting_x,self.starting_y=x,y
                self.img_dup= np.copy(self.img)
        
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.mouse_pressed:
                    self.img_dup = np.copy(self.img)
                    cv2.rectangle(self.img_dup,(self.starting_x,self.starting_y),(x,y),(0,255,0),1)
            #final position of rectange if left mouse button is up then takes the cursor position at ending_x and ending_y 
            elif event == cv2.EVENT_LBUTTONUP:
                self.mouse_pressed=False
                self.ending_x,self.ending_y= x,y

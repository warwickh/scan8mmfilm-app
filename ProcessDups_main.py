from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5 import QtCore, QtGui
from PyQt5 import QtCore, QtWidgets
#from FilmScanModule import Ini, Camera, Frame, Film, getAdjustableRects, getAnalysisTypes
import sys
import shutil
import cv2
import numpy as np 
from resnet50 import ResNet50
from pathlib import Path
import string
from ProcessDups_ui import Ui_MainWindow
#from DupFrameDetector import DupFrameDetector
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from vgg16 import VGG16
from keras.layers import Input
from keras.models import Model

import os
import csv
import glob

class Window(QMainWindow, Ui_MainWindow):
    
    sigToCropTread = pyqtSignal(int)
    sigToScanTread = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.currentLimit = 0.993
        #self.currentLimit = 0.077
        self.resultType = 'similarity'
        self.similarity=0
        self.currentImg = 1
        self.results = []
        self.img1pth = ""
        self.img2pth = ""
        self.img1 = None
        self.img2 = None
        #self.detector = DupFrameDetector()
        image_input = Input(shape=(224, 224, 3))
        model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
        layer_name = 'fc2'
        self.feature_model_vgg = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        self.feature_model_resnet = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
        self.loaded = False
        self.cropFolder = os.path.expanduser("~/scanframes/crop/roll8")
        self.resultPath = os.path.join(self.cropFolder,"dup_results.csv")
        self.connectSignalsSlots()
        self.doLblImagePrep = False
        self.swapTimer = QTimer()
        self.swapTimer.timeout.connect(self.swap)
        self.dsbLimit.setValue(self.currentLimit)
        self.updateInfoPanel()
        QTimer.singleShot(100, self.initData)

    def initData(self, restart=False):
        if restart:
            self.cropFolder=os.path.expanduser("~/scanframes/crop")  
            if not self.selectCropFolder(text="Please select a folder for cropped frames", init=False):
                self.doClose()
                return
        print(f"Starting at {self.cropFolder}")
        self.resultPath = os.path.join(self.cropFolder,"dup_results.csv")
        self.createDupLog(self.cropFolder)

    def selectCropFolder(self, *, text="Select Crop Folder", init=True):
        dir = QtWidgets.QFileDialog.getExistingDirectory(caption=text, directory=os.path.commonpath([self.cropFolder]))
        if dir:
            self.cropFolder = os.path.abspath(dir)
            if init:
                self.updateInfoPanel()
            return True
        return False

    def connectSignalsSlots(self):
        self.pbtnRestart.clicked.connect(self.restart)
        self.pbtnStart.clicked.connect(self.start)
        self.pbtnSwap.clicked.connect(self.swap)
        self.pbtnDup.clicked.connect(self.dup)
        self.pbtnNotDup.clicked.connect(self.notDup)
        self.pbtnSkip.clicked.connect(self.skip)
        self.pbtnDelete.clicked.connect(self.delete)
        self.pbtnRename.clicked.connect(self.rename)
        self.dsbLimit.valueChanged.connect(self.limitChanged)
        self.pbtnDup.setEnabled(False)
        self.pbtnNotDup.setEnabled(False)
        self.pbtnDelete.setEnabled(False)
        self.pbtnRename.setEnabled(True)
        self.pbtnRestart.setEnabled(True)
        self.pbtnStart.setEnabled(False)
        self.pbtnSwap.setEnabled(False)
        self.pbtnSkip.setEnabled(False)

    def updateInfoPanel(self):
        self.lblInfo.setText(f"Current frame {self.currentImg} {self.img2pth} {self.similarity:03f} logged {len(self.results)}")

    def updateDetectInfo(self):
        self.lblInfo.setText(f"{self.currentFrame}/{self.frameCount} {self.currentFrameName} {self.similarity:03f}")

    def limitChanged(self):
        self.currentLimit = self.dsbLimit.value()
        self.updateInfoPanel()

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def start(self):
        print("Starting")
        self.swapTimer.start(500)
        if self.loaded:
            if self.getNextImages():
                self.loadLblImage(self.img1)
                self.updateInfoPanel()
                self.pbtnDup.setEnabled(True)
                self.pbtnNotDup.setEnabled(True)
                self.pbtnDelete.setEnabled(True)
                self.pbtnRestart.setEnabled(True)
                self.pbtnStart.setEnabled(False)
                self.pbtnSwap.setEnabled(True)
                self.pbtnSkip.setEnabled(True)

    def restart(self):
        print("Restarting file")
        self.initData(restart=True)
        open(self.resultPath, 'w').close()
    
    def getCVImage(self, imagePath):
        if not os.path.exists(imagePath):
            print(f"No file {imagePath}")
        else:
            return cv2.imread(imagePath)        

    def getNextRow(self):
        #print(f"loaded {self.loaded}")
        if not self.loaded:
            print("Not loaded")
            return None
        try:
            return next(self.logIterator)
        except:
            return None

    def getNextImages(self):
        while True:
            self.currentRow = self.getNextRow()
            if self.currentRow:
                print(self.currentRow)
                self.img1pth = self.currentRow['img1']
                self.img2pth = self.currentRow['img2']
                self.img1 = self.getCVImage(self.img1pth)
                self.img2 = self.getCVImage(self.img2pth)
                self.similarity = float(self.currentRow[self.resultType])#TODO
                self.updateInfoPanel()
                print(f"{self.img1pth} {self.img2pth} {self.similarity}")
                if self.similarity>=self.currentLimit:
                    return True
            else:
                return False

    def skip(self):
        self.pbtnDup.setEnabled(False)
        self.pbtnNotDup.setEnabled(False)
        self.getNextImages()
        self.loadLblImage(self.img1)
        self.updateInfoPanel()
        self.pbtnDup.setEnabled(True)
        self.pbtnNotDup.setEnabled(True)
        

    def delete(self):
        self.pbtnDelete.setEnabled(False)
        self.deleteAll(self.cropFolder)
        self.pbtnDelete.setEnabled(True)

    def rename(self):
        self.pbtnRename.setEnabled(False)
        self.renameAll(self.cropFolder)
        self.pbtnRename.setEnabled(True)

    def swap(self):
        if self.currentImg==2:
            self.loadLblImage(self.img1)
            self.currentImg=1
        else:
            self.loadLblImage(self.img2)
            self.currentImg=2
        self.updateInfoPanel()
        #print(f"swap to {self.currentImg}")

    def dup(self):
        result = {'img1':self.img1pth, 'img2':self.img2pth, 'similarity': self.similarity, 'isDup': True}
        self.pbtnDup.setEnabled(False)
        self.pbtnNotDup.setEnabled(False)
        self.results.append(result)
        self.saveRow(result)
        if self.getNextImages():
            self.loadLblImage(self.img1)
            self.updateInfoPanel()
            self.pbtnDup.setEnabled(True)
            self.pbtnNotDup.setEnabled(True)

    def notDup(self):
        result = {'img1':self.img1pth, 'img2':self.img2pth, 'similarity': self.similarity, 'isDup': False}
        self.pbtnDup.setEnabled(False)
        self.pbtnNotDup.setEnabled(False)
        self.results.append(result)
        self.saveRow(result)
        if self.getNextImages():
            self.loadLblImage(self.img1)
            self.updateInfoPanel()
            self.pbtnDup.setEnabled(True)
            self.pbtnNotDup.setEnabled(True)
        
    def loadLblImage(self, img):
        if img is not None:
            self.prepLblImage()
            self.lblImage.setPixmap(self.getQPixmap(img,self.scrollAreaWidgetContents))
            self.updateInfoPanel()

    def saveRow(self, rowDict):
        with open(self.resultPath, 'a') as csvfile:
            fieldNames=['img1','img2','similarity','isDup']
            writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
            #writer.writeheader()
            writer.writerow(rowDict)

    def convert_cv_qt(self, cv_img, dest=None):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        if dest is None:
            return QtGui.QPixmap.fromImage(convert_to_Qt_format) 
        else:
            im = convert_to_Qt_format.scaled(dest.width(), dest.height(), QtCore.Qt.KeepAspectRatio)
            return QtGui.QPixmap.fromImage(im)
        
    def getQPixmap(self, img, dest=None):
        return self.convert_cv_qt(img, dest)
        
    def getCropped(self, dest=None):
        ScaleFactor = int(self.ScaleFactor)
        outCrop = cv2.resize(self.imageCropped, (0,0), fx=1/ScaleFactor, fy=1/ScaleFactor)
        return self.convert_cv_qt(outCrop, dest)

    def showImage(self):
        self.prepLblImage()
        self.lblImage.setPixmap(self.frame.getCropOutline(self.scrollAreaWidgetContents) )
        self.lblHoleCrop.setPixmap(self.frame.getHoleCrop())
        #if self.frame.histogram is not None:
        self.lblHist.setPixmap(self.frame.getHistogram())

    def prepLblImage(self):
        if self.doLblImagePrep and self.horizontalLayout_4.SizeConstraint != QtWidgets.QLayout.SetMinimumSize :
            self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
            self.lblImage.clear()
            self.doLblImagePrep = False
 
    def center_crop(self, img, cropAmt):
        width, height = img.shape[1], img.shape[0]
        crop_width = img.shape[1]-cropAmt#dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = img.shape[0]-cropAmt#dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        if cropAmt==400:
            cv2.imwrite(f'./out/chkcrop1.png', img)
            cv2.imwrite(f'./out/chkcrop2.png', crop_img)
        return crop_img 

    def renameAll(self, folder):
        prefix = 'frame'
        os.chdir(folder)
        fileList = sorted(glob.glob(f'{prefix}*.jpg'))
        self.frameCount = len(fileList)
        print(f"Renaming {self.frameCount}")
        self.currentFrame = 0
        for fn in fileList:
            newName = f'{prefix}{self.currentFrame:06d}.jpg'
            print(f"Renaming {fn} to {newName}")
            old = os.path.join(folder,fn)
            new = os.path.join(folder,newName)
            os.rename(old, new)
            self.currentFrame +=1

    def deleteAll(self, folder):
        resultPath = os.path.join(folder,"dup_results.csv")
        archFolder =  os.path.join(folder,"arch")
        os.makedirs(archFolder, exist_ok = True)
        if os.path.exists(resultPath):
            with open(resultPath, 'r') as csvfile:
                fieldNames=['img1','img2','similarity','isDup']
                reader = csv.DictReader(csvfile, fieldnames=fieldNames)
                for row in reader:
                        img2pth = row['img2']
                        isDup = row['isDup'].lower() in ['true']
                        if bool(isDup):
                            print(f"isDup {isDup} so deleting {img2pth}")
                            try:
                                shutil.move(img2pth,os.path.join(archFolder,os.path.basename(img2pth)))
                            except:
                                print(f"Skipping {img2pth}")
                        else:
                            print(f"isDup {isDup} so not deleting {img2pth}")

    def padLocateCrop(self, img1pth, img2pth, cropBoth=True, returnPath=False):
        lbl = "pad"
        ref = cv2.imread(img1pth)
        dy, dx , _= ref.shape
        ref_gray = cv2.cvtColor(self.center_crop(ref,300), cv2.COLOR_BGR2GRAY)
        hr, wr = ref_gray.shape
        ex = cv2.imread(img2pth)
        ex_gray = cv2.cvtColor(self.center_crop(ex,300), cv2.COLOR_BGR2GRAY)
        he, we = ex_gray.shape
        color=(128,128,128)
        wp = we // 2
        hp = he // 2
        ex_gray = cv2.copyMakeBorder(ex_gray, hp,hp,wp,wp, cv2.BORDER_CONSTANT, value=color)
        corrimg = cv2.matchTemplate(ref_gray,ex_gray,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corrimg)
        max_val_corr = '{:.3f}'.format(max_val)
        #print("correlation: " + max_val_corr)
        xx, yy = max_loc
        #print(f'x_match_loc = {xx} x_border {wp} y_match_loc = {yy} y_border {hp}')
        #print(f"Is my movement amount {wp} - {xx} = {wp-xx} and {hp} - {yy} = {hp-yy} ??")
        x_shift = wp-xx
        y_shift = hp-yy
        pad_top = max(0,y_shift)
        pad_bottom = abs(min(0,y_shift))
        pad_left = max(0,x_shift)
        pad_right = abs(min(0,x_shift))
        #print(f"x {x_shift} y {y_shift} top {pad_top} bottom {pad_bottom} left {pad_left} right {pad_right}")
        if cropBoth:
            adjusted_ex = ex[pad_bottom:dy-(2*pad_top)-pad_bottom, pad_right:dx-(2*pad_left)-pad_right]
            adjusted_ref = ref[pad_top:dy-pad_top-(2*pad_bottom), pad_left:dx-pad_left-(2*pad_right)]
            if returnPath:
                cv2.imwrite(f'./tmp1.png', adjusted_ref)
                cv2.imwrite(f'./tmp2.png', adjusted_ex)
                return './tmp1.png', './tmp2.png'
            else:
                return adjusted_ref, adjusted_ex
        else:
            padded_ex = cv2.copyMakeBorder(ex, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
            adjusted_ex = padded_ex[0:dy, 0:dx]
            if returnPath:
                cv2.imwrite(f'./tmp2.png', adjusted_ex)
                return img1pth, './tmp2.png'
            else:
                return img1pth, adjusted_ex

    # avg_pool (AveragePooling2D) output shape: (None, 1, 1, 2048)
    # Latest Keras version causing no 'flatten_1' issue; output shape:(None,2048) 
    def get_feature_vector_fromPIL_vgg(self, img):
        feature_vector = self.feature_model_vgg.predict(img)
        assert(feature_vector.shape == (1,4096))
        return feature_vector

    def get_feature_vector(self,img):#TODO test for direct passing of image
        img1 = cv2.resize(img, (224, 224))
        feature_vector = self.feature_model_vgg.predict(img1.reshape(1, 224, 224, 3))
        return feature_vector

    def calculate_similarity_cosine(self, vector1, vector2):
        #return 1 - distance.cosine(vector1, vector2)
        return cosine_similarity(vector1, vector2)

    def get_similarity_score_vg_c_cv(self, img1, img2):
        image_similarity_cosine = self.calculate_similarity_cosine(self.get_feature_vector(img1), self.get_feature_vector(img2))
        print("VGG16 image similarity_cosine: {:.2f}%".format(image_similarity_cosine[0][0]*100))
        return image_similarity_cosine[0][0]

    def get_similarity_score_vg_c(self, img1pth, img2pth):
        img1 = self.load_image(img1pth)
        img2 = self.load_image(img2pth)
        image_similarity_cosine = self.calculate_similarity_cosine(self.get_feature_vector_fromPIL_vgg(img1), self.get_feature_vector_fromPIL_vgg(img2))
        print("VGG16 image similarity_cosine: {:.2f}%".format(image_similarity_cosine[0][0]*100))
        return image_similarity_cosine[0][0]
    
    def createDupLog(self, folder, force=False):
        os.chdir(folder)
        fileList = sorted(glob.glob('frame*.jpg'))
        self.frameCount = len(fileList)
        lastImage = None
        #logPath = os.path.expanduser(f"~/dups/log.csv")
        self.logPath = os.path.join(folder,f"frame_dups.csv")
        try:
            emptyFile = Path(self.logPath).stat().st_size == 0
        except:
            pass
        self.currentFrame = 0
        if not os.path.exists(self.logPath) or force or emptyFile:
            with open(self.logPath,'w') as logFile:
                for fn in fileList:
                    if True:#fn in ['frame003511.jpg', 'frame003512.jpg', 'frame003513.jpg']:
                        print(f"{fn}")
                        self.currentFrameName = fn
                        currentImage = os.path.join(folder,fn)
                        if lastImage:
                            ref, ex = self.padLocateCrop(lastImage, currentImage, cropBoth=True, returnPath=False)
                            self.similarity = self.get_similarity_score_vg_c_cv(ref, ex)
                            logFile.write(f"{lastImage},{currentImage},{self.similarity}\n")
                            print(f"{lastImage} {currentImage} similarity score_vg_c {self.similarity}")
                            lastImage = currentImage
                        else:
                            lastImage = currentImage
                        self.currentFrame+=1
                        self.updateDetectInfo()
        
        self.getData()

    def getData(self):
        if not os.path.exists(self.logPath):
            print(f"No file {self.logPath}")
            self.loaded = False
        else:
            with open(self.logPath,'r') as logFile:
                headers = ['img1','img2','similarity']
                dict_reader = csv.DictReader(logFile,fieldnames=headers)
                self.logData = list(dict_reader)
            self.logIterator = iter(self.logData)
            self.loaded = True
            self.pbtnStart.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv) 
    win = Window()
    win.show()
    sys.exit(app.exec())
    sys.exit(0)
import numpy as np 
from PIL import Image
from tensorflow.keras.preprocessing import image
import csv
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import matplotlib.animation as animation
import shutil
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity
import os
import glob
import cv2

class DupFrameDetector:
    def __init__(self):
        self.vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))
        self.prepare_model()
        self.loaded = False

    def prepare_model(self):
        for model_layer in self.vgg16.layers:
            model_layer.trainable = False

    def load_image(self, image_path):
        input_image = Image.open(image_path)
        resized_image = input_image.resize((224, 224))
        return resized_image
    
    def get_image_embeddings(self, object_image : image):
        #convert image into 3d array and add additional dimension for model input
        image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
        image_embedding = self.vgg16.predict(image_array)
        return image_embedding
    
    def get_similarity_score(self, first_image : str, second_image : str):
        #Takes image array and computes its embedding using VGG16 model.
        first_image = self.load_image(first_image)
        second_image = self.load_image(second_image)
        first_image_vector = self.get_image_embeddings(first_image)
        second_image_vector = self.get_image_embeddings(second_image)
        similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
        return similarity_score
    
    def combine_images(self, first_image, second_image):
        img1 = cv2.imread(first_image)
        img2 = cv2.imread(second_image)
        h_img = cv2.hconcat([img1, img2])
        return h_img

    #def show_seq(self, first_image,second_image):
    #    fig, ax = plt.subplots()
    #    img1 = [ax.imshow(cv2.imread(first_image), animated=True)]
    #    img2 = [ax.imshow(cv2.imread(second_image), animated=True)]
    #    ims=[img1,img2]
    #    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=0)
    #    plt.show()

    def createDupLog(self, folder, force=False):
        print("createDupLog")
        #roll = "roll6"
        #folder = os.path.expanduser(f"~/scanframes/crop/{roll}")
        os.chdir(folder)
        fileList = sorted(glob.glob('frame*.jpg'))
        lastImage = None
        #logPath = os.path.expanduser(f"~/dups/log.csv")
        self.logPath = os.path.join(folder,f"frame_dups.csv")
        #transPath = os.path.expanduser(f"~/{roll}_transforms.csv")
        if not os.path.exists(self.logPath) or force:
            with open(self.logPath,'w') as logFile:
                for fn in fileList:
                    print(f"{fn}")
                    currentImage = os.path.join(folder,fn)
                    if lastImage:
                        #similarity_score = self.get_similarity_score(lastImage, currentImage)
                        similarity_score = self.get_similarity_score_ssim(lastImage, currentImage)
                        similarity_score_pad = self.get_similarity_score_pad(lastImage, currentImage)
                        logFile.write(f"{lastImage},{currentImage},{similarity_score[0]} {similarity_score_pad[0]}\n")
                        lastImage = currentImage
                        print(f"{lastImage} {currentImage} similarity score {similarity_score[0]}")
                    else:
                        lastImage = currentImage
        self.getData()
        #else:
        #    combine(logPath, transPath)
        #else:
        #   with open(logPath,'r') as logFile:
        #      reader_obj = csv.reader(logFile)
        #      for row in reader_obj: 
        #          print(row)
        #          show_seq(row[0],row[1])

    def center_crop(self, img, cropAmt):
        width, height = img.shape[1], img.shape[0]
        crop_width = img.shape[1]-cropAmt#dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = img.shape[0]-cropAmt#dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        if cropAmt==400:
            cv2.imwrite(f'./out/chkcrop1.jpg', img)
            cv2.imwrite(f'./out/chkcrop2.jpg', crop_img)
        return crop_img

    def get_similarity_score_pad(self, first_image : str, second_image : str):
        lbl = os.path.basename(second_image).split(".")[0]
        ref = self.center_crop(cv2.imread(first_image),300)
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        hr, wr = ref_gray.shape
        ex = self.center_crop(cv2.imread(second_image),300)
        ex_gray = cv2.cvtColor(ex, cv2.COLOR_BGR2GRAY)
        he, we = ex_gray.shape
        color=190
        wp = we // 2
        hp = he // 2
        ex_gray = cv2.copyMakeBorder(ex_gray, hp,hp,wp,wp, cv2.BORDER_CONSTANT, value=color)
        corrimg = cv2.matchTemplate(ref_gray,ex_gray,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corrimg)
        max_val_corr = '{:.3f}'.format(max_val)
        print("correlation: " + max_val_corr)
        xx = max_loc[0]
        yy = max_loc[1]
        print('x_match_loc =',xx,'y_match_loc =',yy)
        # crop the padded example image at top left corner of xx,yy and size hr x wr
        ex_gray_crop = ex_gray[yy:yy+hr, xx:xx+wr]
        ref_grayf = ref_gray.astype(np.float32)
        ex_gray_cropf = ex_gray_crop.astype(np.float32)
        ex_gray_crop = self.center_crop(ex_gray_crop,20)
        ref_gray = self.center_crop(ref_gray,20)
        diff = 255 - np.abs(cv2.add(ref_gray, -ex_gray_crop))
        (score, ssimdiff) = structural_similarity(ref_gray, ex_gray_crop, full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))
        return [score]
    
    def get_similarity_score_ssim(self, first_image : str, second_image : str):
        img1 = cv2.imread(first_image)
        img2 = cv2.imread(second_image)
        img1grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM between the two images
        (score, diff) = structural_similarity(img1grey, img2grey, full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1] 
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        #diff = (diff * 255).astype("uint8")
        #diff_box = cv2.merge([diff, diff, diff])

        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        #thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours = contours[0] if len(contours) == 2 else contours[1]

        #mask = np.zeros(img1.shape, dtype='uint8')
        #filled_after = img2.copy()

        #for c in contours:
        #    area = cv2.contourArea(c)
        #    if area > 40:
        #        x,y,w,h = cv2.boundingRect(c)
        #        cv2.rectangle(img1, (x, y), (x + w, y + h), (36,255,12), 2)
        #        cv2.rectangle(img2, (x, y), (x + w, y + h), (36,255,12), 2)
        #        cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
        #        cv2.drawContours(mask, [c], 0, (255,255,255), -1)
        #        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

        #cv2.imshow('img1', img1)
        #cv2.imshow('img2', img2)
        #cv2.imshow('diff', diff)
        #cv2.imshow('diff_box', diff_box)
        #cv2.imshow('mask', mask)
        #cv2.imshow('filled img2', filled_after)
        #cv2.waitKey()
        
        return [score]

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
                            shutil.move(img2pth,os.path.join(archFolder,os.path.basename(img2pth)))
                        else:
                            print(f"isDup {isDup} so not deleting {img2pth}")
                    
        

    def getNextRow(self):
        #print(f"loaded {self.loaded}")
        if not self.loaded:
            print("Not loaded")
            return None
        return next(self.logIterator)
    
    def getNextImages(self):
        row = self.getNextRow()
        if not row:
            return None, None, None
        #print(row)
        first_image = self.load_image(row['img1'])
        second_image = self.load_image(row['img2'])
        similarity = row['similarity']
        return first_image, second_image, similarity

    def getData(self):
        if not os.path.exists(self.logPath):
            print(f"No file {self.logPath}")
            self.loaded = False
        else:
            with open(self.logPath,'r') as logFile:
                dict_reader = csv.DictReader(logFile,fieldnames=['img1','img2','similarity'])
                self.logData = list(dict_reader)
            self.logIterator = iter(self.logData)
            self.loaded = True

if __name__ == "__main__":
    roll = "roll4"
    folder = os.path.expanduser(f"~/scanframes/crop/{roll}")
    myDetector = DupFrameDetector()
    myDetector.createDupLog(folder)
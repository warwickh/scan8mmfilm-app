import numpy as np 
from PIL import Image
from tensorflow.keras.preprocessing import image
import csv
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import matplotlib.animation as animation

from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
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
                        similarity_score = self.get_similarity_score(lastImage, currentImage)
                        logFile.write(f"{lastImage},{currentImage},{similarity_score[0]} \n")
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
    roll = "roll6"
    folder = os.path.expanduser(f"~/scanframes/crop/{roll}")
    myDetector = DupFrameDetector()
    myDetector.createDupLog(folder)
class DupFrameDetector:
    def __init__(self):
        self.logpath = None



import os
import numpy as np
from resnet50 import ResNet50
#from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from keras.layers import Input
from keras.models import Model
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.spatial import distance

class DupFrameDetector:
    def __init__(self):
        self.logpath = None
        # Use VGG16 model as an image feature extractor 
        image_input = Input(shape=(224, 224, 3))
        model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
        layer_name = 'fc2'
        self.feature_model_vgg = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
        self.feature_model_resnet = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')

    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def process_vgg(self, img1pth, img2pth):
        #vector_VGG16 =get_feature_vector_fromPIL(img_data_list[6])
        # Caculate cosine similarity: [-1,1], that is, [completedly different,same]
        img1 = self.load_image(img1pth)
        img2 = self.load_image(img2pth)
        image_similarity_cosine = self.calculate_similarity_cosine(self.get_feature_vector_fromPIL_vgg(img1), self.get_feature_vector_fromPIL_vgg(img2))
        # Cacluate euclidean similarity: range from [0, 1], that is, [completedly different, same]
        image_similarity_euclidean = self.calculate_similarity_euclidean(self.get_feature_vector_fromPIL_vgg(img1), self.get_feature_vector_fromPIL_vgg(img2))
        print('VGG16 image similarity_euclidean:',image_similarity_euclidean)
        print("VGG16 image similarity_cosine: {:.2f}%".format(image_similarity_cosine[0][0]*100))

    def process_resnet(self, img1pth, img2pth):
        # Cacluate euclidean similarity: range from [0, 1], that is, [completedly different, same]
        img1 = self.load_image(img1pth)
        img2 = self.load_image(img2pth)
        image_similarity_euclidean = self.calculate_similarity_euclidean(self.get_feature_vector_fromPIL(img1), self.get_feature_vector_fromPIL(img2))
        # Caculate cosine similarity: [-1,1], that is, [completedly different, same]
        image_similarity_cosine = self.calculate_similarity_cosine(self.get_feature_vector_fromPIL(img1), self.get_feature_vector_fromPIL(img2))
        print('ResNet50 image similarity_euclidean: ',image_similarity_euclidean)
        print('ResNet50 image similarity_cosine: {:.2f}%'.format(image_similarity_cosine[0][0]*100))

    # avg_pool (AveragePooling2D) output shape: (None, 1, 1, 2048)
    # Latest Keras version causing no 'flatten_1' issue; output shape:(None,2048) 
    def get_feature_vector_fromPIL_vgg(self, img,):
        feature_vector = self.feature_model_vgg.predict(img)
        a, b, c, n = feature_vector.shape
        feature_vector= feature_vector.reshape(b,n)
        return feature_vector

    def get_feature_vector_fromPIL_resnet(self, img,):
        feature_vector = self.feature_model_resnet.predict(img)
        a, b, c, n = feature_vector.shape
        feature_vector= feature_vector.reshape(b,n)
        return feature_vector

    def calculate_similarity_cosine(self, vector1, vector2):
        #return 1 - distance.cosine(vector1, vector2)
        return cosine_similarity(vector1, vector2)

    # This distance can be in range of [0,∞]. And this distance is converted to a [0,1]
    def calculate_similarity_euclidean(self, vector1, vector2):
        return 1/(1 + np.linalg.norm(vector1- vector2))  


    def createDupLog(self, folder, force=False):
        print("createDupLog")
        #roll = "roll6"
        #folder = os.path.expanduser(f"~/scanframes/crop/{roll}")
        os.chdir(folder)
        fileList = sorted(glob.glob('frame*[!_diff].jpg'))
        lastImage = None
        #logPath = os.path.expanduser(f"~/dups/log.csv")
        self.logPath = os.path.join(folder,f"frame_dups.csv")
        #transPath = os.path.expanduser(f"~/{roll}_transforms.csv")
        if True:#not os.path.exists(self.logPath) or force:
            with open(self.logPath,'w') as logFile:
                for fn in fileList:
                    if True:#fn in ['frame003511.jpg', 'frame003512.jpg', 'frame003513.jpg']:
                        print(f"{fn}")
                        currentImage = os.path.join(folder,fn)
                        if lastImage:
                            #similarity_score = self.get_similarity_score(lastImage, currentImage)
                            similarity_score = self.get_similarity_score_greyssim(lastImage, currentImage)
                            logFile.write(f"{lastImage},{currentImage},{similarity_score[0]} \n")
                            print(f"{lastImage} {currentImage} similarity score {similarity_score[0]}")
                            lastImage = currentImage
                        else:
                            lastImage = currentImage
            #self.getData()
                            
if __name__ == "__main__":
    roll = "roll4"
    folder = os.path.expanduser(f"~/scanframes/crop/{roll}")
    myDetector = DupFrameDetector()
    myDetector.createDupLog(folder)
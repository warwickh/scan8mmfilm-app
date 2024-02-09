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
import glob
import numpy as np
from keras.layers import Input
from keras.models import Model
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.spatial import distance
import cv2

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

    def padLocateCrop(self, img1pth, img2pth, cropBoth=True):
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
        print("correlation: " + max_val_corr)
        xx, yy = max_loc
        print(f'x_match_loc = {xx} x_border {wp} y_match_loc = {yy} y_border {hp}')
        print(f"Is my movement amount {wp} - {xx} = {wp-xx} and {hp} - {yy} = {hp-yy} ??")
        x_shift = wp-xx
        y_shift = hp-yy
        pad_top = max(0,y_shift)
        pad_bottom = abs(min(0,y_shift))
        pad_left = max(0,x_shift)
        pad_right = abs(min(0,x_shift))
        print(f"x {x_shift} y {y_shift} top {pad_top} bottom {pad_bottom} left {pad_left} right {pad_right}")
        if cropBoth:
            adjusted_ex = ex[pad_bottom:dy-(2*pad_top)-pad_bottom, pad_right:dx-(2*pad_left)-pad_right]
            adjusted_ref = ref[pad_top:dy-pad_top-(2*pad_bottom), pad_left:dx-pad_left-(2*pad_right)]
            #cv2.imwrite(f'./out/{lbl}_adjusted_ref.png', adjusted_ref)
            cv2.imwrite(f'./tmp1.png', adjusted_ref)
            cv2.imwrite(f'./tmp2.png', adjusted_ex)
            #print(f"input size {hr} {wr} {he} {we} output size {adjusted_ref.shape} {adjusted_ex.shape}")
            #(score_col, ssimdiff_col) = structural_similarity(ref, ex, full=True)
            #8print("Image Similarity Colour: {:.4f}%".format(score_col * 100))
            return './tmp1.png', './tmp2.png'
        else:
            padded_ex = cv2.copyMakeBorder(ex, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
            adjusted_ex = padded_ex[0:dy, 0:dx]
            cv2.imwrite(f'./tmp2.png', adjusted_ex)
            return img1pth, './tmp2.png'

    def get_similarity_score_vg(self, img1pth, img2pth):
        #vector_VGG16 =get_feature_vector_fromPIL(img_data_list[6])
        # Caculate cosine similarity: [-1,1], that is, [completedly different,same]
        img1 = self.load_image(img1pth)
        img2 = self.load_image(img2pth)
        image_similarity_cosine = self.calculate_similarity_cosine(self.get_feature_vector_fromPIL_vgg(img1), self.get_feature_vector_fromPIL_vgg(img2))
        # Cacluate euclidean similarity: range from [0, 1], that is, [completedly different, same]
        image_similarity_euclidean = self.calculate_similarity_euclidean(self.get_feature_vector_fromPIL_vgg(img1), self.get_feature_vector_fromPIL_vgg(img2))
        print('VGG16 image similarity_euclidean:',image_similarity_euclidean)
        print("VGG16 image similarity_cosine: {:.2f}%".format(image_similarity_cosine[0][0]*100))
        return image_similarity_euclidean, image_similarity_cosine[0][0]*100

    def get_similarity_score_rn(self, img1pth, img2pth):
        # Cacluate euclidean similarity: range from [0, 1], that is, [completedly different, same]
        img1 = self.load_image(img1pth)
        img2 = self.load_image(img2pth)
        image_similarity_euclidean = self.calculate_similarity_euclidean(self.get_feature_vector_fromPIL_rn(img1), self.get_feature_vector_fromPIL_rn(img2))
        # Caculate cosine similarity: [-1,1], that is, [completedly different, same]
        image_similarity_cosine = self.calculate_similarity_cosine(self.get_feature_vector_fromPIL_rn(img1), self.get_feature_vector_fromPIL_rn(img2))
        print('ResNet50 image similarity_euclidean: ',image_similarity_euclidean)
        print('ResNet50 image similarity_cosine: {:.2f}%'.format(image_similarity_cosine[0][0]*100))
        return image_similarity_euclidean, image_similarity_cosine[0][0]*100
    
    # avg_pool (AveragePooling2D) output shape: (None, 1, 1, 2048)
    # Latest Keras version causing no 'flatten_1' issue; output shape:(None,2048) 
    def get_feature_vector_fromPIL_vgg(self, img):
        feature_vector = self.feature_model_vgg.predict(img)
        assert(feature_vector.shape == (1,4096))
        return feature_vector

    def get_feature_vector_fromPIL_rn(self, img,):
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

    def createDupLog(self, folder, force=False):
        print("createDupLog")
        #roll = "roll6"
        #folder = os.path.expanduser(f"~/scanframes/crop/{roll}")
        os.chdir(folder)
        fileList = sorted(glob.glob('frame003[56]*[!_diff].jpg'))
        lastImage = None
        #logPath = os.path.expanduser(f"~/dups/log.csv")
        self.logPath = os.path.join(folder,f"frame_dups_vg_rn.csv")
        #transPath = os.path.expanduser(f"~/{roll}_transforms.csv")
        if True:#not os.path.exists(self.logPath) or force:
            with open(self.logPath,'w') as logFile:
                for fn in fileList:
                    if True:#fn in ['frame003511.jpg', 'frame003512.jpg', 'frame003513.jpg']:
                        print(f"{fn}")
                        currentImage = os.path.join(folder,fn)
                        if lastImage:
                            #similarity_score = self.get_similarity_score(lastImage, currentImage)
                            ref, ex = self.padLocateCrop(lastImage, currentImage)
                            similarity_score_vg_e, similarity_score_vg_c = self.get_similarity_score_vg(ref, ex)
                            similarity_score_rn_e, similarity_score_rn_c = self.get_similarity_score_rn(ref, ex)
                            logFile.write(f"{lastImage},{currentImage},{similarity_score_vg_e},{similarity_score_vg_c},{similarity_score_rn_e},{similarity_score_rn_c} \n")
                            print(f"{lastImage} {currentImage} similarity score_vg_e {similarity_score_vg_e} similarity score_vg_c {similarity_score_vg_c} similarity score_rn_e {similarity_score_rn_e} similarity score_rn_c {similarity_score_rn_c}")
                            lastImage = currentImage
                        else:
                            lastImage = currentImage
            #self.getData()
                            
if __name__ == "__main__":
    roll = "roll4"
    folder = os.path.expanduser(f"~/scanframes/crop/{roll}")
    myDetector = DupFrameDetector()
    myDetector.createDupLog(folder)
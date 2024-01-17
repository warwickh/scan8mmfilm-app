import numpy as np 
from PIL import Image
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob

vgg16 = VGG16(weights='imagenet', include_top=False, 
              pooling='max', input_shape=(224, 224, 3))

# print the summary of the model's architecture.
vgg16.summary()

for model_layer in vgg16.layers:
  model_layer.trainable = False

  def load_image(image_path):
    input_image = Image.open(image_path)
    resized_image = input_image.resize((224, 224))
    return resized_image
  
  def get_image_embeddings(object_image : image):
    #convert image into 3d array and add additional dimension for model input
    image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
    image_embedding = vgg16.predict(image_array)
    return image_embedding
  
  def get_similarity_score(first_image : str, second_image : str):
    #Takes image array and computes its embedding using VGG16 model.
    first_image = load_image(first_image)
    second_image = load_image(second_image)
    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)
    similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
    return similarity_score
  
if __name__ == "__main__":
    folder = "C:\\Users\\F98044d\\Downloads\\dup_test"
    #folder = os.path.expanduser("~/scanframes/roll3a")
    os.chdir(folder)
    fileList = sorted(glob.glob('*.jpg'))
    lastImage = None
    for fn in fileList:
        print(f"{fn}")
        currentImage = os.path.join(folder,fn)
        if lastImage:
            similarity_score = get_similarity_score(currentImage, lastImage)
            print(f"similarity score {similarity_score}")
        lastImage = currentImage
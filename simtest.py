import numpy as np 
from PIL import Image
from tensorflow.keras.preprocessing import image
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import cv2

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
  
  def combine_images(first_image, second_image):
    img1 = cv2.imread(first_image)
    img2 = cv2.imread(second_image)
    h_img = cv2.hconcat([img1, img2])
    return h_img


  def show_seq(first_image,second_image):
      fig, ax = plt.subplots()
      img1 = [ax.imshow(cv2.imread(first_image), animated=True)]
      img2 = [ax.imshow(cv2.imread(second_image), animated=True)]
      ims=[img1,img2]
      ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=0)

      # To save the animation, use e.g.
      #
      # ani.save("movie.mp4")
      #
      # or
      #
      # writer = animation.FFMpegWriter(
      #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
      # ani.save("movie.mp4", writer=writer)

      plt.show()


if __name__ == "__main__":
    #folder = "C:\\Users\\F98044d\\Downloads\\dup_test"
    folder = os.path.expanduser("~/scanframes/crop/roll6")
    os.chdir(folder)
    fileList = sorted(glob.glob('*.jpg'))
    lastImage = None
    logPath = os.path.expanduser(f"~/dups/log.csv")
    if not os.path.exists(logPath):
      with open(logPath,'w') as logFile:
        for fn in fileList:
            print(f"{fn}")
            currentImage = os.path.join(folder,fn)
            if lastImage:
                  similarity_score = get_similarity_score(lastImage, currentImage)
                  if similarity_score>0.970:
                      outpath = os.path.expanduser(f"~/dups/{os.path.basename(lastImage)}_{fn}_{float(similarity_score):.04f}.png")
                      cv2.imwrite(outpath, combine_images(lastImage, currentImage))
                      logFile.write(f"{lastImage},{currentImage},{similarity_score} \n")
                  else:
                      lastImage = currentImage
                  print(f"similarity score {similarity_score}")
            else:
              lastImage = currentImage
    else:
       with open(logPath,'r') as logFile:
          reader_obj = csv.reader(logFile)
          for row in reader_obj: 
              print(row)
              show_seq(row[0],row[1])
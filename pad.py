import cv2
import numpy as np
import os
import glob 
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity

class DupFrameDetector:
    def __init__(self):
        self.logpath = None

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

    def get_similarity_score_greyssim(self, first_image : str, second_image : str):
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

    def get_similarity_score_grey(self, first_image : str, second_image : str):
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
        print(f'x_match_loc ={xx} x border {wp} y_match_loc ={yy} y border {hp}')
        # crop the padded example image at top left corner of xx,yy and size hr x wr
        ex_gray_crop = ex_gray[yy:yy+hr, xx:xx+wr]
        ref_grayf = ref_gray.astype(np.float32)
        ex_gray_cropf = ex_gray_crop.astype(np.float32)
        ex_gray_crop = self.center_crop(ex_gray_crop,20)
        ref_gray = self.center_crop(ref_gray,20)
        diff = 255 - np.abs(cv2.add(ref_gray, -ex_gray_crop))
        (score, ssimdiff) = structural_similarity(ref_gray, ex_gray_crop, full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))
        cv2.imwrite(f'./out/{lbl}_ssimdiff.jpg', ssimdiff)
        # compute mean of diff
        print(cv2.mean(diff))
        mean = cv2.mean(diff)[0]
        print("mean of diff in range 0 to 100 =",mean)
        print(f"black {np.sum(diff == 255)}")
        print(f"white {np.sum(diff <255)}")
        print(diff.shape[0]*diff.shape[1])
        th, threshed = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        print(f"contours {len(cnts)}")
        cv2.imwrite(f'./out/{lbl}_threshed.jpg', threshed)
        cv2.imwrite(f'./out/{lbl}_example.jpg', ex)
        cv2.imwrite(f'./out/{lbl}_reference_gray.jpg', ref_gray)
        cv2.imwrite(f'./out/{lbl}_example_gray_padded.jpg', ex_gray)
        cv2.imwrite(f'./out/{lbl}_reference_example_correlation.jpg', (255*corrimg).clip(0,255).astype(np.uint8))
        cv2.imwrite(f'./out/{lbl}_example_gray_padded_cropped.jpg', ex_gray_crop)
        cv2.imwrite(f'./out/{lbl}_diff.jpg', diff)
        return [float((100-mean)/100)]
      
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

if __name__ == "__main__":
    roll = "roll4"
    folder = os.path.expanduser(f"~/scanframes/crop/{roll}")
    myDetector = DupFrameDetector()
    myDetector.createDupLog(folder)
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
            cv2.imwrite(f'./tmp1.png', ref)
            cv2.imwrite(f'./tmp2.png', ex)
            #(score_col, ssimdiff_col) = structural_similarity(ref, ex, full=True)
            #8print("Image Similarity Colour: {:.4f}%".format(score_col * 100))
            return './tmp1.png', './tmp2.png'
        else:
            padded_ex = cv2.copyMakeBorder(ex, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
            adjusted_ex = padded_ex[0:dy, 0:dx]
        #(score_grey, ssimdiff_grey) = structural_similarity(ref_gray, ex_gray_crop, full=True)
        
        #print("Image Similarity Grey: {:.4f}%".format(score_grey * 100))
        
        #cv2.imwrite(f'./out/{lbl}_ref.png', ref)
        #cv2.imwrite(f'./out/{lbl}_ex.png', ex)
        #cv2.imwrite(f'./out/{lbl}_adjusted_ex.png', adjusted_ex)
        

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
        cv2.imwrite(f'./out/{lbl}_ssimdiff.png', ssimdiff)
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
        cv2.imwrite(f'./out/{lbl}_threshed.png', threshed)
        cv2.imwrite(f'./out/{lbl}_example.png', ex)
        cv2.imwrite(f'./out/{lbl}_reference_gray.png', ref_gray)
        cv2.imwrite(f'./out/{lbl}_example_gray_padded.png', ex_gray)
        cv2.imwrite(f'./out/{lbl}_reference_example_correlation.png', (255*corrimg).clip(0,255).astype(np.uint8))
        cv2.imwrite(f'./out/{lbl}_example_gray_padded_cropped.png', ex_gray_crop)
        cv2.imwrite(f'./out/{lbl}_diff.png', diff)
        return [float((100-mean)/100)]
      
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

if __name__ == "__main__":
    roll = "roll4"
    folder = os.path.expanduser(f"~/scanframes/crop/{roll}")
    myDetector = DupFrameDetector()
    #myDetector.createDupLog(folder)
    #myDetector.padLocateCrop("/home/warwickh/scanframes/crop/roll4/frame003607.jpg","/home/warwickh/scanframes/crop/roll4/frame003608.jpg")
    myDetector.padLocateCrop("/home/warwickh/scanframes/crop/roll4/frame003508.jpg","/home/warwickh/scanframes/crop/roll4/frame003509.jpg")
    myDetector.get_similarity_score_grey("/home/warwickh/scanframes/crop/roll4/frame003508.jpg","/home/warwickh/scanframes/crop/roll4/frame003509.jpg")

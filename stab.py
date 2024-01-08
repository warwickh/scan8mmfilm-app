import os
import cv2
import glob
from vidstab import VidStab, layer_overlay, layer_blend
import matplotlib.pyplot as plt

# Initialize object tracker, stabilizer, and video reader
object_tracker = cv2.TrackerCSRT_create()
stabilizer = VidStab()
#vidcap = cv2.VideoCapture("ostrich.mp4")

# Initialize bounding box for drawing rectangle around tracked object
object_bounding_box = None

os.chdir("/home/warwickh/scanframes/crop/roll5a")
        
def layer_custom(foreground, background):
    return layer_blend(foreground, background, foreground_alpha=.8)


def run_stab(fileList):
    for fn in fileList:
        frame = cv2.imread(fn)

        if frame is not None:
            stabilized_frame = stabilizer.stabilize_frame(input_frame=frame,smoothing_window=10,border_size=100,layer_func=layer_overlay)
            #stabilized_frame = stabilizer.stabilize_frame(input_frame=frame,smoothing_window=10,border_size=100,layer_func=layer_blend)
            # Display stabilized output
            cv2.imshow('Frame', stabilized_frame)

            key = cv2.waitKey(5)

def run_graph(fileName):
    stabilizer.stabilize(input_path=fileName, output_path='stable_video.avi')

    stabilizer.plot_trajectory()
    plt.savefig(os.path.expanduser("~/trajectory.png"))
    plt.show()

    stabilizer.plot_transforms()
    plt.savefig(os.path.expanduser("~/transforms.png"))
    plt.show()

if __name__ == "__main__":
    fileList = sorted(glob.glob('*.jpg'))
    #run_stab(fileList)
    run_graph("C:\\Users\\F98044d\\Downloads\\full_long_s8.mp4")
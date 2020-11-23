import numpy as np
import cv2
import imutils
import copy
import os
# import helper_methods as hp
from os import listdir
from os.path import isfile, join

class Tile:
    def __init__(self, image, center):
        self.image = image
        self.center = center


def find_tiles_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0.5)
    edge = cv2.Canny(blur, 100, 220, 3)   
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    indexes = []
    i = 0
    box_list = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        box_list.append(box)
        if area > 5000:
            indexes.append(i)
            cv2.drawContours(frame,[box],0,(0, 60, 255),2)
        i += 1
        
    mask = np.zeros_like(frame) # Create mask where white is what we want, black otherwise
    #centers = []
    tiles = []
    for i in range(len(indexes)):
        card = copy.deepcopy(mask)
        cv2.drawContours(card, box_list, indexes[i], (255, 255, 255), -1) # Draw filled contour in mask
        s = np.where(card == 255)
        x, y = s[1], s[0]
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        card = frame[topy:bottomy+1, topx:bottomx+1]
        xc = (bottomx+topx)/2
        yc = (bottomy+topy)/2
        print("Coordinates center card",i,": x:",xc,"y:",yc)
        tiles.append(Tile(card, (xc,yc)))
        #centers.append((int(xc),int(yc)))
    return tiles;
    # #Draw centers                       
    # for point in centers:
    #     frame = cv2.circle(frame, point, 4, (0,0,255), -1)
    # cv2.imshow("Centers drawn", frame)
    # cv2.waitKey()
    # cv2.destroyAllWindows()



# Global variables
# path to videos
PATH_VIDEOS = 'D:\\Video\'s\\UGent\\Multimedia\\Test_files'
PATH_FRAMES = 'D:\\Video\'s\\UGent\\Multimedia\\Frames'
font = cv2.FONT_HERSHEY_SIMPLEX
prev_values_frames = []
mean_value = 0
saving_file = False
hand_in_frame = False
count = 0
saving_frame_counter = 0
index_video = 6


#Get Path to frames & video files:
video_files = [f for f in listdir(PATH_VIDEOS) if isfile(join(PATH_VIDEOS, f))]
path_to_framesfile = PATH_FRAMES + '\\' + video_files[index_video][0:-4]

#Get frames out of folder:
saved_frames = [f for f in listdir(path_to_framesfile) if isfile(join(path_to_framesfile, f))]
for i,frame in enumerate(saved_frames):
    if i == 1:
        path_frame = path_to_framesfile + '\\' + frame
        frame = cv2.imread(path_frame)
        find_tiles_in_frame(frame)
        

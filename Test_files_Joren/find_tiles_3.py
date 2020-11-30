import numpy as np
import cv2
import imutils
import copy
import os
import helper_methods as hp
from os import listdir
from os.path import isfile, join

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

if not os.path.exists(PATH_FRAMES):
    os.mkdir(PATH_FRAMES)

# Create the backgroundsubtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=0) # The background subtractor will check the last five frames

video_files = [f for f in listdir(PATH_VIDEOS) if isfile(join(PATH_VIDEOS, f))]
print(video_files)

path_to_frames = PATH_FRAMES + '\\' + video_files[index_video][0:-4]

saved_frames = [f for f in listdir(path_to_frames) if isfile(join(path_to_frames, f))]
path_first_frame = path_to_frames + '\\' + saved_frames[1]
frame = cv2.imread(path_first_frame)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0.5)
edge = cv2.Canny(blur, 100, 220, 3)

contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
M = cv2.moments(cnt)
cx = int(M['m10']/M['m00']) # x-center of contour
cy = int(M['m01']/M['m00']) # y-center of contour
area = cv2.contourArea(cnt) # Contour area is given by the function cv2.contourArea() or from moments, M['m00'].
perimeter = cv2.arcLength(cnt,True) # It is also called arc length. It can be found out using cv.arcLength() function. Second argument specify whether shape is a closed contour (if passed True), or just a curve.
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

hull_list = []
indexes = []
i = 0
box_list = []
for cnt in contours:
    # This works but it takes more time    
    # hull = cv2.convexHull(cnt, returnPoints=True)
    # area = cv2.contourArea(hull)
    # print(area)
    # if area > 10000:
    #     hull_list.append(hull)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    area = cv2.contourArea(box)
    box_list.append(box)
    if area > 5000:
        indexes.append(i)
        cv2.drawContours(frame,[box],0,(0, 60, 255),2)
    i += 1

# cv2.drawContours(frame, hull_list, 0, (255, 255, 255), 5)

mask = np.zeros_like(frame) # Create mask where white is what we want, black otherwise
for i in range(len(indexes)):
    card = copy.deepcopy(mask)
    cv2.drawContours(card, box_list, indexes[i], (255, 255, 255), -1) # Draw filled contour in mask
    s = np.where(card == 255)
    x, y = s[1], s[0]
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    card = frame[topy:bottomy+1, topx:bottomx+1]
    cv2.imshow('Out', card)
    cv2.waitKey()
    cv2.destroyAllWindows()
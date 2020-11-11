import numpy as np
import cv2
import imutils
import os
from os import listdir
from os.path import isfile, join

# Global variables
# path to videos
PATH_VIDEOS = 'D:\\Video\'s\\UGent\\Multimedia\\Test_files'
PATH_FRAMES = 'D:\\Video\'s\\UGent\\Multimedia\\Frames'

def handDetected(last):
    if np.mean(last) > 0.0022:
        return False
    else:
        return True

if not os.path.exists(PATH_FRAMES):
    os.mkdir(PATH_FRAMES)

video_files = [f for f in listdir(PATH_VIDEOS) if isfile(join(PATH_VIDEOS, f))]
for index, item in enumerate(video_files):
    video_files[index] = '\\' + item

# Capture the video
video_cap = cv2.VideoCapture(PATH_VIDEOS + video_files[0])

# Read and save the first frame, this is the frame without the cards turned
ret, frame = video_cap.read() # ret is a bool that is true if there is a frame, frame is a part of the video

gray_img = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
edged=cv2.Canny(gray_img,150,255)
contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(frame,contours,-1,(255,0,0),3)
cv2.imshow('Contours in frame', frame)
cv2.waitKey()
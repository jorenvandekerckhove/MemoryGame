import numpy as np
import cv2
import imutils
import os
import helper_methods as hp
from os import listdir
from os.path import isfile, join

# Global variables
# path to videos
PATH_VIDEOS = 'D:\\Video\'s\\UGent\\Multimedia\\Test_files'
PATH_FRAMES = 'D:\\Video\'s\\UGent\\Multimedia\\Frames'

if not os.path.exists(PATH_FRAMES):
    os.mkdir(PATH_FRAMES)

video_files = [f for f in listdir(PATH_VIDEOS) if isfile(join(PATH_VIDEOS, f))]
for index, item in enumerate(video_files):
    video_files[index] = '\\' + item

# Capture the video
video_cap = cv2.VideoCapture(PATH_VIDEOS + video_files[6])

# Create the backgroundsubtractor
background_subtrac = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=0)

# Create the directory for the frames of that video
path_frames_video = PATH_FRAMES + '\\' + video_files[6][0:-4]
if not os.path.exists(path_frames_video):
    os.mkdir(path_frames_video)

hand_in_frame = 0
count = 0
diff_values = []

ret, frame = video_cap.read() # ret is a bool that is true if there is a frame, frame is a part of the video

while ret:

    resized_frame = imutils.resize(frame, width=600)
    mask = background_subtrac.apply(resized_frame)

    # Dilate and erode to clean the fgmask
    # mask = cv2.dilate(mask, (3, 3))
    # mask = cv2.erode(mask, (3, 3))

    # Calculate the amount of pixels in the video
    pixels = mask.shape[0] * mask.shape[1]

    cv2.imshow('Mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = video_cap.read()
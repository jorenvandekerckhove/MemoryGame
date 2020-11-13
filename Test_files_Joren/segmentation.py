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

# Read and save the first frame, this is the frame without the cards turned
ret, frame = video_cap.read() # ret is a bool that is true if there is a frame, frame is a part of the video

# cv2.imwrite(path_frames_video + video_files[0][0:-4] + '_'+ str(count) + '.png', frame)
control_frame = frame

while ret:
    count += 1
    ret, frame = video_cap.read()
    if count > 10:
        hp.show_live_feed(frame)        
        diff_values.append(hp.find_diferences(frame, control_frame))
        count = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# print(diff_values)
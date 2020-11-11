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
video_cap = cv2.VideoCapture(PATH_VIDEOS + video_files[0])

# Create the backgroundsubtractor
background_subtrac = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=0)

# Create the directory for the frames of that video
path_frames_video = PATH_FRAMES + '\\' + video_files[0][0:-4]
if not os.path.exists(path_frames_video):
    os.mkdir(path_frames_video)

last = np.array([])
hand_in_frame = 0
count = 0
total = 0
saved = False
hand = False

# Read and save the first frame, this is the frame without the cards turned
ret, frame = video_cap.read() # ret is a bool that is true if there is a frame, frame is a part of the video

cv2.imwrite(path_frames_video + video_files[0][0:-4] + '_'+ str(count) + '.png', frame)
ret, frame = video_cap.read()
hp.draw_contours(frame, (200, 255))

while ret:
    count += 1
    # Je gebruikt de resize function van imutils voor uw foto te verkleinen
    # resized_frame = imutils.resize(frame, width=600)
    # Of gebruik de resize function van cv2
    scale_percent = 50 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim)

    # Applying the background subtractor
    mask = background_subtrac.apply(resized_frame)

    # Dilate and erode to clean the mask
    mask = cv2.dilate(mask, (3, 3))
    mask = cv2.erode(mask, (3, 3))

    # Calculate the amount of pixels in the video
    pixels = mask.shape[0] * mask.shape[1]

    if count > 20:
        som = np.sum(mask) / pixels / 255
        if len(last) < 20:
            last = np.append(last, som)
        else:
            last = np.delete(last, 0)
            last = np.append(last, som)

        # Calculate mean of last 20 fgmasks
        # If mean difference between the frames is larger than given threshold, a hand is detected
        # A frame has two images with the right side up if a hand has gone out of the frame any two times
        if handDetected(last):
            hand = True
            if saved == False:
                hand_in_frame += 1
                saved = True
                if hand_in_frame % 2 == 0:
                    cv2.imwrite(path_frames_video + video_files[0][0:-4] + '_'+ str(total) + '.png', frame)
                    total += 1
        else:
            hand = False
            saved = False

    ret, frame = video_cap.read()

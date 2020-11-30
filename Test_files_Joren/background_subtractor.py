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
font = cv2.FONT_HERSHEY_SIMPLEX
prev_values_frames = []
mean_value = 0
saving_file = False
hand_in_frame = False
count = 0
saving_frame_counter = 0 # This variable is used so that we don't need to save all the frames, but only the frames where tiles are rotated
index_video = 6

if not os.path.exists(PATH_FRAMES):
    os.mkdir(PATH_FRAMES)

video_files = [f for f in listdir(PATH_VIDEOS) if isfile(join(PATH_VIDEOS, f))]
for index, item in enumerate(video_files):
    video_files[index] = '\\' + item

# Capture the video
video_cap = cv2.VideoCapture(PATH_VIDEOS + video_files[index_video])

# Create the backgroundsubtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=0) # The background subtractor will check the last five frames

# Create the directory for the frames of that video
path_frames_video = PATH_FRAMES + '\\' + video_files[index_video][0:-4]
if not os.path.exists(path_frames_video):
    os.mkdir(path_frames_video)

ret, frame = video_cap.read() # ret is a bool that is true if there is a frame, frame is a part of the video
cv2.imwrite(path_frames_video + video_files[index_video][0:-4] + '_'+ str(saving_frame_counter) + '.png', frame)
saving_frame_counter += 1
while ret:
    count += 1
    resized_frame = imutils.resize(frame, width=600)
    foreground_mask = backSub.apply(resized_frame)

    # Calculate the amount of pixels in the video
    pixels = foreground_mask.shape[0] * foreground_mask.shape[1]
    
    if count > 20: # this is used so that the foreground mask can be formed, otherwise it will say that it already found a hand
        summation = np.sum(foreground_mask) / pixels
        if len(prev_values_frames) <= 10:
            prev_values_frames.append(summation)
        else:
            prev_values_frames.pop(0)
            prev_values_frames.append(summation)
        mean_value = np.mean(prev_values_frames)
        if mean_value > 0.45:
            hand_in_frame = True             
        else:
            hand_in_frame = False
        
        if hand_in_frame:
            if not saving_file:
                saving_file = True                               

        if saving_file and not hand_in_frame:
            if saving_frame_counter % 2 != 0:
                cv2.imwrite(path_frames_video + video_files[index_video][0:-4] + '_'+ str(saving_frame_counter) + '.png', frame)
                saving_frame_counter += 1
            else:
                saving_frame_counter += 1
            saving_file = False

    gray = np.zeros((50, foreground_mask.shape[1]), np.uint8)
    gray[:] = 150
    vcat1 = cv2.vconcat((gray, foreground_mask))    
    cv2.putText(vcat1,f"Hand detected: {hand_in_frame}", (30,30), font, 1,(0,0,0), 2, 0)
    cv2.imshow('Mask', vcat1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = video_cap.read()

video_cap.release()
cv2.destroyAllWindows()
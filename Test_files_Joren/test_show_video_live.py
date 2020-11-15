import numpy as np
import cv2
import imutils
import os
import helper_methods as hp
from matplotlib import pyplot as plt
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

while(video_cap.isOpened()):
    ret, frame = video_cap.read()
    resized_frame = imutils.resize(frame, width=600)
    gray_img = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)
    edged=cv2.Canny(gray_img, 20,255)
    hp.draw_contours(resized_frame, (200, 255))

    black = [0, 0, 0]   
    font = cv2.FONT_HERSHEY_SIMPLEX

    constant = cv2.copyMakeBorder(resized_frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)
    gray= np.zeros((50, constant.shape[1], 3), np.uint8)
    gray[:] = (150, 150, 150)
    vcat1 = cv2.vconcat((gray, constant))    
    cv2.putText(vcat1,'Video',(30,50), font, 2,(0,0,0), 3, 0)
    
    constant = cv2.copyMakeBorder(edged, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)
    gray = np.zeros((50, constant.shape[1]), np.uint8)
    gray[:] = 150
    vcat2 = cv2.vconcat((gray, constant))
    vcat2 = cv2.cvtColor(vcat2, cv2.COLOR_GRAY2BGR)
    cv2.putText(vcat2,'Edging applied',(30,50), font, 2,(0,0,0), 3, 0)

    final_img = cv2.hconcat((vcat1, vcat2))

    cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
    cv2.imshow('Final', final_img)    
    # cv2.resizeWindow('Final', (vcat2.shape[1]//2,final_img.shape[0]//2))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
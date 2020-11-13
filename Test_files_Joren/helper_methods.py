import cv2
import numpy as np
import imutils
from skimage.measure import compare_ssim

def draw_contours(frame, treshold):
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edged=cv2.Canny(gray_img,treshold[0], treshold[1])
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(frame,contours,-1,(255,0,0),1)

def show_live_feed(frame):
    resized_frame = imutils.resize(frame, width=600)
    gray_img = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)
    edged=cv2.Canny(gray_img, 20,255)
    draw_contours(resized_frame, (200, 255))

    black = [0, 0, 0]   
    font = cv2.FONT_HERSHEY_SIMPLEX

    constant = cv2.copyMakeBorder(resized_frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)
    gray= np.zeros((50, constant.shape[1], 3), np.uint8)
    gray[:] = (150, 150, 150)
    vcat1 = cv2.vconcat((gray, constant))    
    cv2.putText(vcat1,'Video', (30,30), font, 1,(0,0,0), 2, 0)
    
    constant = cv2.copyMakeBorder(edged, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)
    gray = np.zeros((50, constant.shape[1]), np.uint8)
    gray[:] = 150
    vcat2 = cv2.vconcat((gray, constant))
    vcat2 = cv2.cvtColor(vcat2, cv2.COLOR_GRAY2BGR)
    cv2.putText(vcat2,'Edging applied', (30,30), font, 1,(0,0,0), 2, 0)

    final_img = cv2.hconcat((vcat1, vcat2))

    cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
    cv2.imshow('Final', final_img)    
    # cv2.resizeWindow('Final', (vcat2.shape[1]//2,final_img.shape[0]//2))

def find_diferences(current, control):
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    control_gray = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(control_gray, current_gray, full=True)
    diff = (diff * 255).astype("uint8")
    return score
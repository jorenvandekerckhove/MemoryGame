import cv2

def draw_contours(frame, treshold):
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edged=cv2.Canny(gray_img,treshold[0], treshold[1])
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(frame,contours,-1,(255,0,0),1)
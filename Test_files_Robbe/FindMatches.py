import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import os

dirpath = os.path.dirname(os.path.abspath(__file__))

def find_matches_in_grid_and_label(grid):
    imagelist = grid.grid_tiles

imagelist= []
for i in range(1,8):
    imagelist.append(cv2.imread(dirpath+"\images\\QueryImages\\" + str(i) + ".PNG"))
    
queryimage = cv2.imread(dirpath+"\\images\\QueryImages\\1.PNG",0)


orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
orblist =[]
matcheslist = []

# queryimage = cv2.Canny(queryimage,100,200)
kp2, des2 = orb.detectAndCompute(queryimage, None)

for image in imagelist:
    # image = cv2.Canny(image,100,200)
    kp, des = orb.detectAndCompute(image, None)
    orblist.append((kp,des))
    
    matches = bf.match(des, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    matcheslist.append(matches)
    
    #matches_img = cv2.drawMatches(image,kp,queryimage,kp2,matches, outImg=None, flags=2)
    print(len(matches))
    #plt.imshow(matches_img)
    plt.show()
pass
    
    

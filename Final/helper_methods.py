import cv2
import numpy as np
import imutils
import copy
import os
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from skimage.measure import compare_ssim
from tile import Tile
from grid import Grid
from skimage.metrics import structural_similarity as ssim

def calcMSE(image1, image2): #Calc MSE of greyscale --> Higher value = less similar
    resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    err_matrix = (image1.astype("float") - resized.astype("float"))**2
    err = np.sum(err_matrix)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def calcSSIM(image1, image2): #Calc SSIM of greyscale --> Higher value = more similar
    return ssim(image1,image2, multichannel=True)

def calcMI(image1, image2, bins = 20):
    resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    hgram, x_edges, y_edges = np.histogram2d(image1.ravel(), resized.ravel(), bins)

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def draw_contours(frame, treshold):
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur_gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    edged=cv2.Canny(blur_gray_img,treshold[0], treshold[1])
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame,contours,-1,(160,160,50), 2)

def show_live_feed(frame, foreground_mask, hand_in_frame, mean_value):
    resized_frame = imutils.resize(frame, width=600)
    # gray_img = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)
    # blur_gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    # edged=cv2.Canny(blur_gray_img, 50,100)
    # draw_contours(resized_frame, (50, 180))

    black = [0, 0, 0]   
    font = cv2.FONT_HERSHEY_DUPLEX 

    constant = cv2.copyMakeBorder(resized_frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)
    gray= np.zeros((50, constant.shape[1], 3), np.uint8)
    gray[:] = (150, 150, 150)
    vcat1 = cv2.vconcat((gray, constant))    
    cv2.putText(vcat1,'Video',(30,40), font, 1,(0,0,0), 3, 0)
    
    resized_foreground_mask = cv2.resize(foreground_mask, (resized_frame.shape[1], resized_frame.shape[0]))
    constant = cv2.copyMakeBorder(resized_foreground_mask, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)
    gray = np.zeros((50, constant.shape[1]), np.uint8)
    gray[:] = 150
    vcat2 = cv2.vconcat((gray, constant))
    vcat2 = cv2.cvtColor(vcat2, cv2.COLOR_GRAY2BGR)
    cv2.putText(vcat2,f"Hand detected: {hand_in_frame}",(30,40), font, 1,(0,0,0), 3, 0)

    final_img = cv2.hconcat((vcat1, vcat2))

    # cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
    cv2.imshow('Final', final_img)    
    # cv2.resizeWindow('Final', (vcat2.shape[1]//2,final_img.shape[0]//2))

def create_grid(frame, contours):
    colored_frame = copy.deepcopy(frame)
    boxes = []
    width_tile = 0
    height_tile = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if area > 5000:
            width_tile += rect[1][0]
            height_tile += rect[1][1]
            boxes.append(box)
            cv2.drawContours(colored_frame,[box],0,(0, 60, 255),2)
    
    width_tile = width_tile//len(boxes)
    height_tile = height_tile//len(boxes)

    mask = np.zeros_like(colored_frame)
    card = copy.deepcopy(mask)
    cv2.drawContours(card, boxes, -1, (255, 255, 255), -1) # Draw filled contour in mask
    s = np.where(card == 255)
    x, y = s[1], s[0]
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    card = frame[topy:bottomy, topx:bottomx]
    grid = Grid(topx, topy, bottomx, bottomy, width_tile, height_tile, None)
    return grid

def get_template_tile(frame, contours):
    box = None
    colored_frame = copy.deepcopy(frame)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if area > 5000:
            tile_to_show = box
            cv2.drawContours(colored_frame,[box],0,(0, 60, 255),2)
            break
    
    mask = np.zeros_like(colored_frame)
    card = copy.deepcopy(mask)
    cv2.drawContours(card, [box], -1, (255, 255, 255), -1) # Draw filled contour in mask
    s = np.where(card == 255)
    x, y = s[1], s[0]
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    card = frame[topy:bottomy, topx:bottomx]
    # resized_frame = imutils.resize(card, width=600)
    return card

def find_tiles_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0.5)
    edge = cv2.Canny(blur, 100, 220, 3)   
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colored_frame = copy.deepcopy(frame)
    indexes = []
    i = 0
    box_list = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        box_list.append(box)
        if area > 8000:
            indexes.append(i)
            cv2.drawContours(colored_frame,[box],0,(0, 60, 255),2)
        i += 1
    
    # resized_frame = imutils.resize(frame, width=600)
    # cv2.imshow('Testing', resized_frame)
    # cv2.waitKey()
    mask = np.zeros_like(colored_frame) # Create mask where white is what we want, black otherwise
    tiles = []
    for i in range(len(indexes)):
        card = copy.deepcopy(mask)
        cv2.drawContours(card, box_list, indexes[i], (255, 255, 255), -1) # Draw filled contour in mask
        s = np.where(card == 255)
        x, y = s[1], s[0]
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        card = frame[topy:bottomy+1, topx:bottomx+1]
        xc = (bottomx+topx)/2
        yc = (bottomy+topy)/2
        tiles.append(Tile(card, (xc,yc)))
    return tiles

def find_place_in_grid(grid, tile, divide_x, divide_y):
    # print('Grid min x:', grid.min_x, 'max x:', grid.max_x, 'min y:', grid.min_y, 'max y:', grid.max_y)
    # print('tile x:', tile.center[0], 'y:', tile.center[1])
    offsetx = (grid.max_x - grid.min_x)/(divide_x)
    offsety = (grid.max_y - grid.min_y)/(divide_y)
    # print('offsetx:',offsetx)
    # print('offsety:',offsety)
    x_min_off, x_max_off = grid.min_x, grid.min_x + offsetx 
    y_min_off, y_max_off = grid.min_y, grid.min_y + offsety
    x_pos, y_pos = 0, 0
    for i in range(len(grid.grid_array[0])):
        if x_min_off <= tile.center[0] and tile.center[0] <= x_max_off:
            x_pos = i
            break
        else:
            x_min_off += offsetx
            x_max_off += offsetx
    
    # print('x_min_off:',x_min_off)
    # print('x_max_off:',x_max_off)
    # print()

    for i in range(len(grid.grid_array)):
        if y_min_off <= tile.center[1] and tile.center[1] <= y_max_off:
            y_pos = i
            break
        else:
            y_min_off += offsety
            y_max_off += offsety
    
    # print('y_min_off:',y_min_off)
    # print('y_max_off:',y_max_off)
    # print()

    return x_pos, y_pos

def find_matches_in_grid_and_label(grid):
    imagelist = grid.grid_array
    height = len(imagelist)
    width = len(imagelist[0])
    total_size = height*width
    
    matcheslist = np.empty((height,width,total_size))
    solution = np.zeros((height,width),dtype=int)
    match_amount_max = np.zeros(total_size, dtype=int) #Keep track of what position has been matched already
    
    orb = cv2.ORB_create(WTA_K = 3)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    for y, queryrow in enumerate(imagelist):
        for x, queryimage in enumerate(queryrow):
            kp2, des2 = orb.detectAndCompute(queryimage, None)
            # plt.imshow(queryimage)
            # plt.show()
            k = x + width*y
            for i, row in enumerate(imagelist):
                for j, image in enumerate(row):
                    kp, des = orb.detectAndCompute(image, None)
                    matches = bf.match(des, des2)
                    # print(i,j,k)
                    matcheslist[i][j][k] = len(matches)

    #Stable marriage problem possible solution??
    i=0
    pairing_number = 0
    while np.count_nonzero(match_amount_max) != len(match_amount_max) and i < total_size:
        if(match_amount_max[i] == 0):
            temp = matcheslist[:,:,i]
            idx = np.argpartition(temp, temp.size - 2, axis=None)[-2:]
            res = np.column_stack(np.unravel_index(idx,temp.shape))
            match_amount = temp[res[1][0]][res[1][1]]
            if(match_amount > match_amount_max[width*res[1][0] + res[1][1]]):
                match_amount_max[idx[0]] = match_amount
                match_amount_max[idx[1]] = match_amount
                for position in res:
                    solution[position[0]][position[1]] = pairing_number
            pairing_number +=1
        i +=1
    return solution
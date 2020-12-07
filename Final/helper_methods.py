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
    resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return ssim(image1,resized)

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

def show_live_feed(frame):
    resized_frame = imutils.resize(frame, width=600)
    gray_img = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2GRAY)
    blur_gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    edged=cv2.Canny(blur_gray_img, 50,100)
    draw_contours(resized_frame, (50, 180))

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

def create_grid(frame, contours):
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
            cv2.drawContours(frame,[box],0,(0, 60, 255),2)
    
    width_tile = width_tile//len(boxes)
    height_tile = height_tile//len(boxes)

    mask = np.zeros_like(frame)
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
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if area > 5000:
            tile_to_show = box
            cv2.drawContours(frame,[box],0,(0, 60, 255),2)
            break
    
    mask = np.zeros_like(frame)
    card = copy.deepcopy(mask)
    cv2.drawContours(card, [box], -1, (255, 255, 255), -1) # Draw filled contour in mask
    s = np.where(card == 255)
    x, y = s[1], s[0]
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    card = frame[topy+5:bottomy-5, topx+5:bottomx-5]
    resized_frame = imutils.resize(card, width=600)
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
        if area > 5000:
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

def find_place_in_grid(grid, tile):
    # print('Grid min x:', grid.min_x, 'max x:', grid.max_x, 'min y:', grid.min_y, 'max y:', grid.max_y)
    # print('tile x:', tile.center[0], 'y:', tile.center[1])
    offsetx = (grid.max_x - grid.min_x)/3
    offsety = (grid.max_y - grid.min_y)/3
    x_min_off, x_max_off = 0, offsetx 
    y_min_off, y_max_off = 0, offsety
    x_pos, y_pos = 0, 0
    for i in range(len(grid.grid_array[0])):
        if x_min_off <= tile.center[0] and tile.center[0] <= x_max_off:
            x_pos = i
            break
        else:
            x_min_off += offsetx
            x_max_off += offsetx
    
    for i in range(len(grid.grid_array)):
        if y_min_off <= tile.center[1] and tile.center[1] <= y_max_off:
            y_pos = i
            break
        else:
            y_min_off += offsety
            y_max_off += offsety
    
    return x_pos, y_pos
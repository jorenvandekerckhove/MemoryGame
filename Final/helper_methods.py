import cv2
import numpy as np
import imutils
import copy
import os
from os import listdir
from os.path import isfile, join
from tile import Tile
from grid import Grid

def draw_contours(frame, treshold):
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur_gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    edged=cv2.Canny(blur_gray_img,treshold[0], treshold[1])
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame,contours,-1,(160,160,50), 2)

def show_live_feed(frame, foreground_mask, hand_in_frame, mean_value):
    resized_frame = imutils.resize(frame, width=600)

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

    cv2.imshow('Live video', final_img)    

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
    offsetx = (grid.max_x - grid.min_x)/(divide_x)
    offsety = (grid.max_y - grid.min_y)/(divide_y)
   
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
  
    for i in range(len(grid.grid_array)):
        if y_min_off <= tile.center[1] and tile.center[1] <= y_max_off:
            y_pos = i
            break
        else:
            y_min_off += offsety
            y_max_off += offsety

    return x_pos, y_pos

def find_matches_in_grid_and_label(grid, wta_k=3,border_orb=31, border_img=0):
    imagelist = grid.grid_array
    height = len(imagelist)
    width = len(imagelist[0])
    total_size = height*width
    
    matcheslist = np.empty((height,width,total_size))
    solution = np.zeros((height,width),dtype=int)
    match_amount_max = np.zeros(total_size, dtype=int) #Keep track of what position has been matched already
    
    orb = cv2.ORB_create(WTA_K = wta_k, edgeThreshold=border_orb, patchSize=border_orb) # original value 3
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    for y, queryrow in enumerate(imagelist):
        for x, queryimage in enumerate(queryrow):
            kp2, des2 = orb.detectAndCompute(queryimage[border_img:queryimage.shape[1]-border_img, border_img:queryimage.shape[0]-border_img, :], None)
            k = x + width*y
            for i, row in enumerate(imagelist):
                for j, image in enumerate(row):
                    kp, des = orb.detectAndCompute(image[border_img:image.shape[1]-border_img, border_img:image.shape[0]-border_img, :], None)
                    matches = bf.match(des, des2)
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
        
    for i,row in enumerate(imagelist):
        for j,image in enumerate(row):
            rect = 255 * np.ones(shape=[50, 50, 3], dtype=np.uint8)
            if len(str(solution[i][j])) >= 2:
                cv2.rectangle(image, (5,0), (55,35), (255, 255, 255), -1)
            else:
                cv2.rectangle(image, (5,0), (35,35), (255, 255, 255), -1)
            cv2.putText(image, str(solution[i][j]), (10,27), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    # solution_image = np.zeros((int(grid.width_tile*2), int(grid.height_tile*(total_size/2)), 3))
    # for i in range(total_size):
    #     newimage = []
    #     locations = np.asarray(np.where(solution == i)).T
    #     print(locations)
    #     if(locations.size != 0):
    #         image1 = imagelist[locations[0][0]][locations[0][1]] 
    #         image2 = imagelist[locations[1][0]][locations[1][1]]
    #         h1, w1 = image1.shape[:2]
    #         h2, w2 = image2.shape[:2]
    #         newimage = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    #         newimage[:h1, :w1, :3] = image1
    #         newimage[:h2,w1:w1+w2,:3] =  image2
    #         newimage = newimage.reshape(solution_image.shape[0], int(solution_image.shape[1]/(total_size/2)), 3)
    #         solution_image = np.concatenate((newimage, solution_image), axis=0)
    # cv2.imshow("Please werk",solution_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
                               
    return solution
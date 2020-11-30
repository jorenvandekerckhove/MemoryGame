import numpy as np
import cv2
import imutils
import copy
import os
import helper_methods as hp
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

class Tile:
    def __init__(self, image, center):
        self.image = image
        self.center = center

class Grid:
    def __init__(self, min_x, min_y, max_x, max_y, width_tile, height_tile, grid_array):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.width_tile = width_tile
        self.height_tile = height_tile
        self.grid_array = grid_array

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
            cv2.drawContours(frame,[box],0,(0, 60, 255),2)
        i += 1
        
    mask = np.zeros_like(frame) # Create mask where white is what we want, black otherwise
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

# Global variables
# path to videos
PATH_VIDEOS = 'D:\\Video\'s\\UGent\\Multimedia\\Test_files'
PATH_FRAMES = 'D:\\Video\'s\\UGent\\Multimedia\\Frames'
index_video = 6
grid_array = []
threshold = 60

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

if not os.path.exists(PATH_FRAMES):
    os.mkdir(PATH_FRAMES)

video_files = [f for f in listdir(PATH_VIDEOS) if isfile(join(PATH_VIDEOS, f))]
# print(video_files)

path_to_frames = PATH_FRAMES + '\\' + video_files[index_video][0:-4]

saved_frames = [f for f in listdir(path_to_frames) if isfile(join(path_to_frames, f))]
path_to_frame = path_to_frames + '\\' + saved_frames[0]
frame = cv2.imread(path_to_frame)

# First we find the template for a 'not-rotated' tile
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0.5)
edge = cv2.Canny(blur, 100, 220, 3)
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

grid = create_grid(frame, contours) # Build the grid with the properties get from the first frame
template_tile = get_template_tile(frame, contours) # Get the template of a 'non-rotated' tile

for i in range(4):
    grid_array.append([])
    for j in range(4):
        grid_array[i].append(template_tile)

grid.grid_array = grid_array

kp2, des2 = orb.detectAndCompute(template_tile, None)
max_width = 0 # Used for later
max_height = 0 # Used for later
for i in range(1, len(saved_frames)):
    path_to_frame = path_to_frames + '\\' + saved_frames[i]
    frame = cv2.imread(path_to_frame)
    tiles = find_tiles_in_frame(frame)
    
    for tile in tiles:
        if tile.image.shape[0] > max_height:
            max_height = tile.image.shape[0]
        if tile.image.shape[1] > max_width:
            max_width = tile.image.shape[1]

        kp, des = orb.detectAndCompute(tile.image, None)
        
        matches = bf.match(des, des2)

        if len(matches) < threshold:
            xpos, ypos = find_place_in_grid(grid, tile)
            grid.grid_array[xpos][ypos] = tile.image

# Creating an image of the grid, so where all the tiles are shown
final_image = np.zeros((max_height*4,max_width*4,3), dtype=np.uint8)
current_x = 0
current_y = 0
for i in range(4):
    current_x = 0
    for j in range(4):
        final_image[current_y:(grid.grid_array[i][j].shape[0]+current_y), current_x:(grid.grid_array[i][j].shape[1]+current_x),:] = grid.grid_array[i][j]
        current_x += grid.grid_array[i][j].shape[1]
    current_y += max_height

cv2.imwrite('Example_grid.png', final_image)
import numpy as np
import cv2
import imutils
import argparse
import os
import copy
import helper_methods as hp
from os import listdir
from os.path import isfile, join
from tile import Tile
from grid import Grid
from skimage.measure import compare_ssim

font = cv2.FONT_HERSHEY_SIMPLEX
prev_values_frames = []
mean_value = 0
saving_file = False
hand_in_frame = False
count = 0
saving_frame_counter = 0 # This variable is used so that we don't need to save all the frames, but only the frames where tiles are rotated

parser = argparse.ArgumentParser(description='Process some values.')
parser.add_argument('--PATH_VIDEOS', type=str, default='D:\\Video\'s\\UGent\\Multimedia\\Test_files', help='Path to the videos')
parser.add_argument('--PATH_FRAMES', type=str, default='D:\\Video\'s\\UGent\\Multimedia\\Frames', help='Path where the frames are gonna be saved')
parser.add_argument('--INDEX_VIDEO', type=int, default=6, help='Choose which movie it needs to play')
parser.add_argument('--THRESHOLD_HAND', type=float, default=0.45, help='Used for checking if the hand is in screen')
parser.add_argument('--THRESHOLD_MATCHES', type=float, default=20000000, help='Used for checking if the tile is not the template tile')
parser.add_argument('--COLS', type=int, default=4, help='Give the total of cols of the board')
parser.add_argument('--ROWS', type=int, default=4, help='Give the total of rows of the board')
parser.add_argument('--PLAY', type=int, default=0, help='Play the video or just check the frames')

args = parser.parse_args()

print(' \n\
        ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗     ██████╗  █████╗ ███╗   ███╗███████╗ \n\
        ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝    ██╔════╝ ██╔══██╗████╗ ████║██╔════╝ \n\
        ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝     ██║  ███╗███████║██╔████╔██║█████╗   \n\
        ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝      ██║   ██║██╔══██║██║╚██╔╝██║██╔══╝   \n\
        ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║       ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗ \n\
        ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝        ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝')

if not os.path.exists(args.PATH_FRAMES):
    os.mkdir(args.PATH_FRAMES)

video_files = [f for f in listdir(args.PATH_VIDEOS) if isfile(join(args.PATH_VIDEOS, f))]
for index, item in enumerate(video_files):
    video_files[index] = '\\' + item

# Capture the video
video_cap = cv2.VideoCapture(args.PATH_VIDEOS + video_files[args.INDEX_VIDEO])

# Create the backgroundsubtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=0) # The background subtractor will check the last five frames

# Create the directory for the frames of that video
path_frames_video = args.PATH_FRAMES + '\\' + video_files[args.INDEX_VIDEO][0:-4]
if not os.path.exists(path_frames_video):
    os.mkdir(path_frames_video)

if args.PLAY == 1:
    ret, frame = video_cap.read() # ret is a bool that is true if there is a frame, frame is a part of the video
    cv2.imwrite(path_frames_video + video_files[args.INDEX_VIDEO][0:-4] + '_'+ str(saving_frame_counter) + '.png', frame)
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
            if mean_value > args.THRESHOLD_HAND: # check if the hand is in screen, so that we know if the frame needs to be saved
                hand_in_frame = True             
            else:
                hand_in_frame = False
            
            if hand_in_frame:
                if not saving_file:
                    saving_file = True                               

            if saving_file and not hand_in_frame:
                if saving_frame_counter % 2 != 0: # It happens that when tiles are rotated, that nothing is shown to the user, so we save only the frames where the rotated tiles contains a picture
                    cv2.imwrite(path_frames_video + video_files[args.INDEX_VIDEO][0:-4] + '_'+ str(saving_frame_counter) + '.png', frame)
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

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

path_to_frames = args.PATH_FRAMES + '\\' + video_files[args.INDEX_VIDEO][0:-4]
saved_frames = [f for f in listdir(path_to_frames) if isfile(join(path_to_frames, f))]
path_to_frame = path_to_frames + '\\' + saved_frames[0]
frame = cv2.imread(path_to_frame)
grid_array = []

# First we find the template for a 'not-rotated' tile
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0.5)
edge = cv2.Canny(blur, 100, 220, 3)
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

grid = hp.create_grid(frame, contours) # Build the grid with the properties get from the first frame
template_tile = hp.get_template_tile(frame, contours) # Get the template of a 'non-rotated' tile

for i in range(args.ROWS):
    grid_array.append([])
    for j in range(args.COLS):
        grid_array[i].append(template_tile)

grid.grid_array = grid_array

# cv2.imshow('Template_tile', template_tile)
# cv2.waitKey()

# kp2, des2 = orb.detectAndCompute(template_tile, None)
max_width = 0 # Used for later
max_height = 0 # Used for later

# avg_color_per_row = np.average(template_tile, axis=0)
# avg_colors = np.average(avg_color_per_row, axis=0)
# int_averages_template = np.array(avg_colors, dtype=np.uint8)
# print(int_averages_template)

# print(grid.min_x)
# print(grid.min_y)
# print(grid.max_x)
# print(grid.max_y)
# print(grid.width_tile)
# print(grid.height_tile)

for i in range(1, len(saved_frames)):
    # print(saved_frames)
    # print(path_to_frame)
    path_to_frame = path_to_frames + '\\' + saved_frames[i]
    frame = cv2.imread(path_to_frame)
    tiles = hp.find_tiles_in_frame(frame)

    # resized_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    # cv2.imshow('Frame', resized_frame)
    # cv2.waitKey()
    for tile in tiles:
        if tile.image.shape[0] > max_height:
            max_height = tile.image.shape[0]
        if tile.image.shape[1] > max_width:
            max_width = tile.image.shape[1]

        # Method 1: feature detection
        # kp, des = orb.detectAndCompute(tile.image, None)        
        # matches = bf.match(des, des2)

        # Method 2: average color
        # avg_color_per_row = np.average(tile.image, axis=0)
        # avg_colors = np.average(avg_color_per_row, axis=0)
        # int_averages_tile = np.array(avg_colors, dtype=np.uint8)

        # Method 3: SSIM
        # (score, diff) = compare_ssim(cv2.cvtColor(template_tile, cv2.COLOR_BGR2GRAY), cv2.resize(cv2.cvtColor(tile.image, cv2.COLOR_BGR2GRAY), (template_tile.shape[1], template_tile.shape[0])) , full=True)
        # diff = (diff * 255).astype("uint8")
        # print(score)

        # Method 4: template matching
        # print('New tile')
        resized = cv2.resize(tile.image, (template_tile.shape[1], template_tile.shape[0]))
        res = cv2.matchTemplate(resized,template_tile, cv2.TM_CCOEFF)
        # print(res[0][0])
        
        # cv2.imshow('Template', template_tile)
        # cv2.imshow('Tile', tile.image)
        # cv2.waitKey()
        if res[0][0] < args.THRESHOLD_MATCHES:   
            # print('Tile center:', tile.center)
            xpos, ypos = hp.find_place_in_grid(grid, tile, args.COLS, args.ROWS)
            # print(xpos, ypos)
            # cv2.imshow('Tile', tile.image)
            # cv2.waitKey()  
            grid.grid_array[ypos][xpos] = tile.image

# Creating an image of the grid, so where all the tiles are shown
final_image = np.zeros((max_height*args.ROWS,max_width*args.COLS,3), dtype=np.uint8)
current_x = 0
current_y = 0
for i in range(args.ROWS):
    current_x = 0
    for j in range(args.COLS):
        final_image[current_y:(grid.grid_array[i][j].shape[0]+current_y), current_x:(grid.grid_array[i][j].shape[1]+current_x),:] = grid.grid_array[i][j]
        current_x += grid.grid_array[i][j].shape[1]
    current_y += max_height

cv2.imwrite('Example_grid.png', final_image)
print('Finished')
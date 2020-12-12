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
parser.add_argument('--INDEX_VIDEO', type=int, default=4, help='Choose which movie it needs to play')
parser.add_argument('--THRESHOLD_HAND', type=float, default=0.25, help='Used for checking if the hand is in screen')
parser.add_argument('--THRESHOLD_SUB', type=float, default=32, help='Used for checking if the tile is not the template tile')
parser.add_argument('--THRESHOLD_MATCHES', type=float, default=0.05, help='Used for checking if the tile is not the template tile')
parser.add_argument('--COLS', type=int, default=10, help='Give the total of cols of the board')
parser.add_argument('--ROWS', type=int, default=7, help='Give the total of rows of the board')
parser.add_argument('--BORDER', type=int, default=10, help='Give the total of rows and cols that need to dissapear for calculation')
parser.add_argument('--METHOD', type=str, default="hist", help='Choose which method you want to use. sub = subtraction, sub_key = subtraction or keypoints')
parser.add_argument('--PLAY', type=int, default=0, help='Play the video or just check the frames')
parser.add_argument('--SAVE', type=int, default=1, help='0 for saving not all the frames, 1 for saving all the frames')

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

mean_value = 0
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
                if args.SAVE == 1:
                    cv2.imwrite(path_frames_video + video_files[args.INDEX_VIDEO][0:-4] + '_'+ str(saving_frame_counter) + '.png', frame)
                    saving_frame_counter += 1
                    saving_file = False
                else:
                    if saving_frame_counter % 2 != 0: # It happens that when tiles are rotated, that nothing is shown to the user, so we save only the frames where the rotated tiles contains a picture
                        cv2.imwrite(path_frames_video + video_files[args.INDEX_VIDEO][0:-4] + '_'+ str(saving_frame_counter) + '.png', frame)
                        saving_frame_counter += 1
                    else:
                        saving_frame_counter += 1
                    saving_file = False

        # gray = np.zeros((50, foreground_mask.shape[1]), np.uint8)
        # gray[:] = 150
        # vcat1 = cv2.vconcat((gray, foreground_mask))    
        # cv2.putText(vcat1,f"Hand detected: {hand_in_frame}, val: {mean_value}", (30,30), font, 1,(0,0,0), 2, 0)
        # cv2.imshow('Mask', vcat1)
        hp.show_live_feed(frame, foreground_mask, hand_in_frame, mean_value)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = video_cap.read()

    video_cap.release()
    cv2.destroyAllWindows()

orb = cv2.ORB_create()
bf = cv2.BFMatcher()
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

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

max_width = 0 # Used for later
max_height = 0 # Used for later

for i in range(1, len(saved_frames)):
    print("Looking at frame", i, ":", saved_frames[i])
    path_to_frame = path_to_frames + '\\' + saved_frames[i]
    frame = cv2.imread(path_to_frame)
    tiles = hp.find_tiles_in_frame(frame)
    validated_files = 0
    print('Total tiles found:', len(tiles))

    for tile in tiles:
        if tile.image.shape[0] > max_height:
            max_height = tile.image.shape[0]
        if tile.image.shape[1] > max_width:
            max_width = tile.image.shape[1]

        resized = cv2.resize(tile.image, (template_tile.shape[1], template_tile.shape[0]))
        
        if args.METHOD == "match":
            kp2, des2 = orb.detectAndCompute(template_tile[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], None)
            kp, des = orb.detectAndCompute(resized[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], None)        
            
            # Matches normal
            # matches = bf.match(des,des2)
            # matches = sorted(matches, key = lambda x:x.distance)
            # Matches with ratio test
            matches = bf.knnMatch(des, des2, k=2)        
            good = []
            for m,n in matches:
                if m.distance < 0.8*n.distance:
                    good.append([m])
            total_matches = len(good)

            if total_matches <= args.THRESHOLD_MATCHES:
                xpos, ypos = hp.find_place_in_grid(grid, tile, args.COLS, args.ROWS) 
                grid.grid_array[ypos][xpos] = tile.image

        elif args.METHOD == "hist":
            hsv_base = cv2.cvtColor(template_tile[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], cv2.COLOR_BGR2HSV)
            hsv_test = cv2.cvtColor(resized[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], cv2.COLOR_BGR2HSV)
            
            h_bins = 50
            s_bins = 60
            histSize = [h_bins, s_bins]
            # hue varies from 0 to 179, saturation from 0 to 255
            h_ranges = [0, 180]
            s_ranges = [0, 256]
            ranges = h_ranges + s_ranges # concat lists
            # Use the 0-th and 1-st channels
            channels = [0, 1]

            hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            hist_test = cv2.calcHist([hsv_test], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            base_test = cv2.compareHist(hist_base, hist_test, 0)

            resized = resized
            if base_test < args.THRESHOLD_MATCHES:
                validated_files += 1
                xpos, ypos = hp.find_place_in_grid(grid, tile, args.COLS, args.ROWS) 
                grid.grid_array[ypos][xpos] = tile.image

        elif args.METHOD == "template":
            threshold = 0.8
            template_logo = template_tile[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:]
            res_template_1 = cv2.matchTemplate(resized,template_logo, cv2.TM_CCOEFF_NORMED)
            loc1 = np.where(res_template_1 >= threshold)

            template_logo = imutils.rotate_bound(template_logo, 270)
            res_template_2 = cv2.matchTemplate(resized,template_logo, cv2.TM_CCOEFF_NORMED)        
            loc2 = np.where(res_template_2 >= threshold)

            resized = resized
            if len(loc1[0])==0 and len(loc1[1])==0 and len(loc2[1])==0 and len(loc2[1])==0 and res_template_1.max() < args.THRESHOLD_MATCHES and res_template_1.min() < 0 and res_template_2.max() < args.THRESHOLD_MATCHES and res_template_2.min() < 0:
                validated_files += 1
                xpos, ypos = hp.find_place_in_grid(grid, tile, args.COLS, args.ROWS) 
                grid.grid_array[ypos][xpos] = tile.image

        elif args.METHOD == "sub":
            res = cv2.subtract(template_tile[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], resized[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:])
            res = np.mean(res)
            resized = resized
            if res > args.THRESHOLD_SUB:   
                validated_files += 1
                xpos, ypos = hp.find_place_in_grid(grid, tile, args.COLS, args.ROWS) 
                grid.grid_array[ypos][xpos] = tile.image

        elif args.METHOD == "key":
            kp, des = orb.detectAndCompute(resized[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], None)        
            resized = resized
            if len(kp) > args.THRESHOLD_MATCHES:
                validated_files += 1
                xpos, ypos = hp.find_place_in_grid(grid, tile, args.COLS, args.ROWS)
                grid.grid_array[ypos][xpos] = tile.image

        elif args.METHOD == "sub_key":
            kp, des = orb.detectAndCompute(resized[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], None)        
            res = cv2.subtract(template_tile[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], resized[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:])
            res = np.mean(res)
            resized = resized
            if res > args.THRESHOLD_SUB or len(kp) > args.THRESHOLD_MATCHES:   
                # print('Tile center:', tile.center)
                validated_files += 1
                xpos, ypos = hp.find_place_in_grid(grid, tile, args.COLS, args.ROWS)
                grid.grid_array[ypos][xpos] = tile.image
        
        elif args.METHOD == "sub_and_key":
            kp, des = orb.detectAndCompute(resized[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], None)        
            res = cv2.subtract(template_tile[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:], resized[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:])
            res = np.mean(res)
            small = resized[args.BORDER:template_tile.shape[1]-args.BORDER, args.BORDER:template_tile.shape[0]-args.BORDER,:] 
            if res > args.THRESHOLD_SUB and len(kp) > args.THRESHOLD_MATCHES:   
                validated_files += 1
                xpos, ypos = hp.find_place_in_grid(grid, tile, args.COLS, args.ROWS)
                grid.grid_array[ypos][xpos] = tile.image

    print('Validated tiles found:', validated_files)
    print()

solution = hp.find_matches_in_grid_and_label(grid)
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
print(solution)
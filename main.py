#!/usr/bin/env python3

import os, sys, shutil
import cv2
import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt
from moviepy.editor import *

PHOTOS_DIR = '__photos__/'
if PHOTOS_DIR[:-1] not in os.listdir():
    os.mkdir(PHOTOS_DIR)

def homography(P1, P2):
    """
    Compute the homography matrix H given two sets of corresponding points
    This allows us to transform the input video frames to a normalized front-
    facing view of the whiteboard.
    Input:
        - P1: The corner points of the whiteboard in the input image
        - P2: The desired location of the transformed coordinates
    Output:
        - H: Homography matrix
    """
    A = []
    b = []
    for p1, p2 in zip(P1, P2):
        A.extend([
            [p1[0], p1[1], 1, 0, 0, 0, -p1[0] * p2[0], -p1[1] * p2[0]],
            [0, 0, 0, p1[0], p1[1], 1, -p1[0] * p2[1], -p1[1] * p2[1]]
        ])
        b.extend(p2)

    A = np.array(A)
    b = np.array(b)
    h = np.append(spla.solve(A, b), 1)
    H = np.reshape(h, (3, 3))

    return H

def getVideoStreams(left='Videos/left.MOV', right='Videos/right.MOV'):
    """
    Get the Video Streams using OpenCV2's VideoCapture class
    Input:
        - left: the filename of the video taken from the lefthand side
                Default: 'Videos/left.MOV'
        - right: the filename of the video taken from the righthand side
                Default: 'Videos/right.MOV'

    Output:
        - lcap: the Video Stream of the lefthand side
        - rcap: the Video Stream of the righthand side
    """
    lcap = cv2.VideoCapture(left)
    rcap = cv2.VideoCapture(right)
    return lcap, rcap

def getCorners(lframe, rframe):
    """
    Get the points corresponding to the left frame and the right frame
    If the preset configurations are not there, then plot the 4 corners of the
    video. Else, load these preset configurations.
    Input:
        - lframe: the 1st frame on which to plot the points
        - rframe: the 2nd frame on which to plot the points
    Output:
        - left_pts: Points plotted for the left frame
        - right_pts: Points plotted for the right frame
    """
    
    if 'whiteboard_left.npy' not in os.listdir() or \
       'whiteboard_right.npy' not in os.listdir():
        print("Plot the 4 corners of the whiteboard")
        print("In order of Top Left, Top Right, Bottom Right, Bottom Left")
        print("Click on the image to continue")
        plt.imshow(lframe)
        left_pts = plt.ginput(4)

        print("Plot the 4 corners of the whiteboard")
        print("In order of Top Left, Top Right, Bottom Right, Bottom Left")
        plt.imshow(rframe)
        right_pts = plt.ginput(5)[1:]

        plt.close()
        np.save('whiteboard_left.npy', left_pts)
        np.save('whiteboard_right.npy', right_pts)
    else:
        left_pts = np.load('whiteboard_left.npy')
        right_pts = np.load('whiteboard_right.npy')

    return left_pts, right_pts

def processInput(lcap, rcap):
    """
    Process the two different videos and concatenate them into one video.
    Because OpenCV2 requires a lot of work when writing a new video, it is best
    to use MoviePy given the input frames.

    Input:
        - lcap: the Video Stream from the lefthand side
        - rcap: the Video Stream from the righthand side
    Output:
        
    """
    # Assumption: left_pts = [top left, top right, bottom right, bottom left]
    retl, lframe = lcap.read()
    retr, rframe = rcap.read()
    
    center_pts = [
        (0, 0),
        (lframe.shape[1], 0),
        (lframe.shape[1], lframe.shape[0]),
        (0, lframe.shape[0])
    ]

    Hl = homography(left_pts, center_pts)
    Hr = homography(right_pts, center_pts)
    
    frames = []
    counter = 0
    while lcap.isOpened() and rcap.isOpened():
        print(f"Frame #{counter} ... ".ljust(20), end='', flush=True)
        retl, lframe = lcap.read()
        retr, rframe = rcap.read()
        
        if retl and retr:
            lframe = cv2.warpPerspective(lframe, Hl, dsize=(lframe.shape[1], lframe.shape[0]))
            rframe = cv2.warpPerspective(rframe, Hr, dsize=(lframe.shape[1], rframe.shape[0]))
            frame = (0.5 * lframe.astype(np.float) + 0.5 * rframe.astype(np.float)).astype(np.uint8)
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            cv2.imwrite(PHOTOS_DIR+f"{counter}.jpg", frame)
            frames.append(PHOTOS_DIR+f"{counter}.jpg")
            print("+")
        else:
            break
        counter += 1

    lcap.release()
    rcap.release()
    cv2.destroyAllWindows()

    clips = [ImageClip(m).set_duration(1/30) for m in frames]
    video = concatenate_videoclips(clips, method='compose')
    video.write_videofile('output.mp4', fps=30.0)
    shutil.rmtree(PHOTOS_DIR)

# Demo the code
if __name__ == '__main__':
    lcap, rcap = getVideoStreams()

    if not lcap.isOpened() or not rcap.isOpened():
        print("Error opening video stream or file")
        sys.exit()

    retl, lframe = lcap.read()
    retr, rframe = rcap.read()
    left_pts, right_pts = getCorners(lframe, rframe)

    offsetr = 20

    for i in range(offsetr):
        retr, rframe = rcap.read()

    processInput(lcap, rcap)
    print("Completed! Your video is now ready to be viewed!")
    print(f"Your video is located at '{os.getcwd() + '/'}output.mp4'")

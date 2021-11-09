import cv2
import time
import math
import numpy as np
import argparse
NS_PER_S=1000000000
ESC=27
#https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV.git
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
net = cv2.dnn.readNet(r"c:\Documents and Settings\rpell\Coding\Fasto\EAST-Detector-for-text-detection-using-OpenCV\frozen_east_text_detection.pb")

def find_square(oframe):
    frame = cv2.cvtColor(oframe, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(frame, threshold1=50, threshold2=200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80, minLineLength=10, maxLineGap=10);
    for line in lines[0]:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        cv2.line(oframe, pt1, pt2, (0, 0, 255), 3)

def find_text(frame):
    pass

def show_live():
    t0=time.time()

    cv2.namedWindow("camera")
    print(time.time()-t0)
    cam = cv2.VideoCapture(0)
    print(time.time()-t0)

    if cam.isOpened(): # try to get the first frame
        print(time.time()-t0)
        rval, frame = cam.read()
        print(time.time()-t0)
        orig = frame.copy()
        (H, W) = frame.shape[:2]
        print("h",H,"w",W)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        print(scores, geometry);
        cv2.imshow("camera", frame)
    else:
        rval = False

    while True:
        key = cv2.waitKey(20)
        if key == ESC: # exit on ESC
            break
    cam.release()
    cv2.destroyWindow("camera")

def foo():
    t1=time.time_ns()
    frames=0
    s='0'
    while rval:
        frames+=1
        dt=time.time_ns()-t1
        if (dt > NS_PER_S):
            s="fps:" + str(frames)
            t1=time.time_ns()
            frames=0
        cv2.putText(frame, s, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, lineType=cv2.LINE_AA)
        find_square(frame)
        cv2.imshow("camera", frame)
        rval, frame = cam.read()

        key = cv2.waitKey(20)
        if key == ESC: # exit on ESC
            break

    cam.release()
    cv2.destroyWindow("camera")

if __name__ == '__main__':
    show_live()
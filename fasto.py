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

# Reference
# https://github.com/PyImageSearch/imutils/blob/master/imutils/object_detection.py#L4
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")
    
def find_square(oframe):
    frame = cv2.cvtColor(oframe, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(frame, threshold1=50, threshold2=200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80, minLineLength=10, maxLineGap=10);
    for line in lines[0]:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        cv2.line(oframe, pt1, pt2, (0, 0, 255), 3)

def find_text(frame,H,W,rH,rW):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    #print(blob.shape)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    '''
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        min_confidence=0.2
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            #print(scoresData[x])

    #np.save(r'rects.data', rects, allow_pickle=True, fix_imports=True)
    rects=non_max_suppression(np.array(rects))
    for (startX, startY, endX, endY) in rects:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    '''
    return frame

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
        #orig = frame.copy()
        (H, W) = frame.shape[:2]
        rW = 1. #W / float(newW)
        rH = 1. #H / float(newH)

        print("h",H,"w",W)

        #print(scores, geometry);
        #blobb = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3], 1) 
        #cv2.imshow("camera", blobb)

        #cv2.imshow("camera", frame)
        cv2.imshow("camera", frame)

    else:
        rval = False

    '''#-----
    while True:
        key = cv2.waitKey(20)
        if key == ESC: # exit on ESC
            break
    cam.release()
    cv2.destroyWindow("camera")

def foo():
    cam = cv2.VideoCapture(0)
    #-----'''
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
        #find_square(frame)
        frame=find_text(frame,H,W,rH,rW)
        cv2.imshow("camera", frame)
        rval, frame = cam.read()

        key = cv2.waitKey(20)
        if key == ESC: # exit on ESC
            break

    cam.release()
    cv2.destroyWindow("camera")

def bounding_reduction():
    height=480
    width=640
    cv2.namedWindow("camera")
    frame = np.zeros((height, width, 3), np.uint8)
    rects=np.load(r'c:\Documents and Settings\rpell\Coding\Fasto\rects.data.npy')
    print(rects.shape)
    for i, (startX, startY, endX, endY) in enumerate(rects):
        print(i, startX,startY,endX,endY)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    rects=non_max_suppression(rects)
    print("----------")
    print(rects.shape)
    for i, (startX, startY, endX, endY) in enumerate(rects):
        print(i, startX,startY,endX,endY)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("camera", frame)
    while True:
        key = cv2.waitKey(20)
        if key == ESC: # exit on ESC
            break

    cv2.destroyWindow("camera")


if __name__ == '__main__':
    show_live()
    #bounding_reduction()
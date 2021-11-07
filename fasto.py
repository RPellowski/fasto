import cv2
import time
import math
NS_PER_S=1000000000
ESC=27

def find_square(oframe):
    frame = cv2.cvtColor(oframe, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(frame, threshold1=50, threshold2=200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80, minLineLength=10, maxLineGap=10);
    for line in lines[0]:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        cv2.line(oframe, pt1, pt2, (0, 0, 255), 3)

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
    else:
        rval = False

    t1=time.time_ns()
    frames=0
    s='0'
    while rval:
        frames+=1
        dt=time.time_ns()-t1
        if (dt > NS_PER_S):
            s=str(frames)
            t1=time.time_ns()
            frames=0
        cv2.putText(frame, s, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
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
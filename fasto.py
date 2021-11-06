import cv2
import time

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

    while rval:
        cv2.imshow("camera", frame)
        rval, frame = cam.read()
        cv2.putText(frame, "My text", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cam.release()
    cv2.destroyWindow("camera")

if __name__ == '__main__':
    show_live()
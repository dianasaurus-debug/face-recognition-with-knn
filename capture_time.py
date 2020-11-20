import cv2
import time
import dlib
import os

capture = cv2.VideoCapture(0)
total=0
newpath = r'D:\\trainwithdlib\\dataset\\test\\'
capture.set(3, 640)
capture.set(4, 480)
img_counter = 0
frame_set = []
start_time = time.time()
color_green = (0,255,0)
line_width = 3
detector = dlib.get_frontal_face_detector()
while True:
    ret, frame = capture.read()
    key = cv2.waitKey(1)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    for det in dets:
        cv2.rectangle(frame,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
    cv2.imshow('frame', frame)
    if key == 27: #esc untuk tutup
        break
    if time.time() - start_time >= 5: #kalo udah 3 detik dicapture
        total+=1
        cv2.imshow("Capturing", frame)
        img_ = cv2.resize(frame[det.top():det.bottom(), det.left():det.right()],(100,100))
        cv2.imwrite(newpath + str(img_counter) + ".jpg", img_)
        img_name = str(img_counter) + ".jpg"
        cv2.waitKey(1650)
        print("{} disimpan!".format(img_name))
        cv2.destroyWindow('Capturing')
        start_time = time.time()
    if total == 3 :
        break
    img_counter += 1
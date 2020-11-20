import face_recognition
import cv2
import matplotlib.pyplot as plt
import dlib
img_path1 = '1.jpeg'
img_path2 = '2.png'
color_green = (0,255,0)
line_width = 3
image1 = face_recognition.load_image_file(img_path1)
image2 = face_recognition.load_image_file(img_path2)
face_bounding_boxes1 = face_recognition.face_locations(image1)
face_bounding_boxes2 = face_recognition.face_locations(image2)
print("Gambar 1 \n: {} ".format(face_recognition.face_encodings(image1, known_face_locations=face_bounding_boxes1)[0]))
print("Gambar 2 \n: {} ".format(face_recognition.face_encodings(image2, known_face_locations=face_bounding_boxes2)[0]))
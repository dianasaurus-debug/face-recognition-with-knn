import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import matplotlib.pyplot as plt
images = []
directory=os.listdir('dataset/train')
for each in directory:
    plt.figure()
    currentFolder = 'dataset/train/' + each
    plt.rcParams.update({'figure.max_open_warning': 0})
    for i, file in enumerate(os.listdir(currentFolder)):
        fullpath = currentFolder + "/" + file
        print(fullpath)
        img=plt.imread(fullpath)
        images.append(img)
for i, image in enumerate(images) :
    plt.subplot((78/6)+1, 6, i+1)
    plt.axis('off')
    plt.imshow(image)
plt.show()
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Gambar {} tidak sesuai: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Menemukan lebih dari 1 wajah"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
                print("Wajah terdeteksi untuk NRP : {}".format(class_dir))
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

if __name__ == "__main__":
    print("Training dataset ...  ")
    classifier = train("dataset/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
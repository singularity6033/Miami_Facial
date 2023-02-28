# import the necessary packages
import urllib.request
import pandas as pd
from tqdm import tqdm
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os
import csv

image_path = './images/photos'
save_coordinates_x, save_coordinates_y = 'result_x.csv', 'result_y.csv'

# change to 81 face points will cover the forehead, just include the dataset path in here
shape_predictor_path = './dataset/shape_predictor_81_face_landmarks.dat'

# plan a --> read images directly
filenames = os.listdir(image_path)

# print(filenames)

# plan b --> read urls
# img_urls_dir = ''
# file = open(img_urls_dir)
# urls_list = list(csv.reader(file))
# file.close()

for ii, filename in tqdm(enumerate(filenames)):
    felon_id = filename.lstrip("['").rstrip("'].jpg")
    img_dir = os.path.join(image_path, filename)

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # load the input image, resize it, and convert it to grayscale
    try:
        image = cv2.imread(img_dir)

        # plan b
        # req = urllib.request.urlopen('https://images.weserv.nl/?url=' + url)
        # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # image = cv2.imdecode(arr, -1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)
        landmarks_x, landmarks_y = list(), list()
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then convert the facial landmark (x, y) to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            idx = 0
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.275, (0, 255, 0), 1)
                landmarks_x.append(x)
                landmarks_y.append(y)
                idx += 1
        cv2.imwrite('./results/' + str(felon_id) + '.png', image)

        landmarks_x.append(felon_id)
        landmarks_y.append(felon_id)
        rx = [landmarks_x[::-1]]
        ry = [landmarks_y[::-1]]

        scx = open(save_coordinates_x, 'a+', newline='')
        with scx:
            w1 = csv.writer(scx)
            w1.writerows(rx)
        scx.close()

        scy = open(save_coordinates_y, 'a+', newline='')
        with scy:
            w2 = csv.writer(scy)
            w2.writerows(ry)
        scy.close()
    except:
        continue

import cv2
from utils import calculateLBPVector
import numpy as np
import pickle
#test_im = cv2.imread("test/1316.jpg")
#test_data = calculateLBPVector(test_im)
#test_2d = np.reshape(test_data, (-1, 768)) #Reshaping 768x1 matrix to 1x768 TODO: SOLVED A LOT!!!

model = pickle.load(open("model/lbp_model.svm", 'rb'))

def normalizeHistogramArray(historgram_array):
    predict_2d = np.reshape(historgram_array, (-1, 768))
    return  predict_2d

import cv2

cap = cv2.VideoCapture(0)

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    if ret:

        faces_rects = haar_cascade_face.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_image = frame[y:y + h, x:x + w]
            crop_image = cv2.resize(crop_image, (150,150))
            lbp_hist = calculateLBPVector(crop_image)
            final_histogram = normalizeHistogramArray(lbp_hist)
            result = model.predict(final_histogram)

            if np.array2string(result) == "['real']":
                result = "fake"
            else:
                result = "real"

            cv2.putText(frame,result,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)

        cv2.imshow('frame',frame)
        cv2.waitKey(0)
 
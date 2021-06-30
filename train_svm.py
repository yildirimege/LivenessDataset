import os
from utils import calculateLBPVector
from sklearn import svm
from sklearn.metrics import classification_report
from imutils import paths
import cv2
import time
import numpy as np
import pickle

training_dir =  list(paths.list_images("Dataset"))

datas = []
labels = []



def loadDatas():
    for imagePath in training_dir:
        label = list(imagePath.split(os.path.sep))[-2]
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (150,150))
        print(imagePath)
        hist = calculateLBPVector(image)

        datas.append(hist)
        labels.append(label)




'''print("Initialising has been started.")
start = time.process_time()
#loadDatas()
print("*********************************************************************")
print(f"Time Taken: {time.process_time() - start}")
print(f"Size of Datas: {len(datas)}") 
print("*********************************************************************")
print(f"Size of Labels: {len(labels)}")
print("*********************************************************************")
print(f"Data of first Element: {datas[0]}")
print("*********************************************************************")
print(f"Label of first Element: {labels[0]}")
print("*********************************************************************")'''

loadDatas()

datas = np.array(datas)
labels = np.array(labels)

np_labels = np.array(labels)

nsamples, nx, ny = datas.shape
d2_datas = datas.reshape((nsamples, nx*ny))




model = svm.SVC(kernel="rbf", verbose=True, C=1, gamma="scale")
model.fit(d2_datas, np_labels)

filename = 'model/lbp_model.svm'
pickle.dump(model, open(filename, 'wb+'))
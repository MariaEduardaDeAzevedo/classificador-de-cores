import cv2 as cv
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import warnings

def load_dataset():
    dataset = pd.read_csv("dataset.csv")
    hist = list()

    for path in dataset["CAMINHO"]:
        image = cv.imread(path)
        blue = cv.calcHist([image], [0], None, [256], (0,256)).flatten()
        green = cv.calcHist([image], [1], None, [256], (0,256)).flatten()
        red = cv.calcHist([image], [2], None, [256], (0,256)).flatten()
        hist.append(np.array(list(blue) + list(green) + list(red)))
    
    dataset["HIST"] = hist

    return dataset

def pca(X_train):
    return PCA(n_components=90).fit(list(X_train))


def model(pca, X_train, y_train):
    parametros = {
        "n_neighbors": [2,3,5,7,11,13]
    }

    knn = GridSearchCV(KNeighborsClassifier(), param_grid=parametros)

    train = pca.transform(list(X_train))

    knn.fit(train, y_train)

    return knn
import cv2 as cv
import model
from sklearn.model_selection import train_test_split
import numpy as np

cam = cv.VideoCapture(0)

running = True

dataset = model.load_dataset()
X = dataset["HIST"]
y = dataset["CLASSE"]
X_train, _ , y_train, _ = train_test_split(X, y, train_size=0.8, random_state=13)

pca = model.pca(X_train)
knn = model.model(pca, X_train, y_train)

dict_classes = {
    0: ("AMARELO", (0,255,255)),
    1: ("VERMELHO", (0,0,255)),
    2: ("AZUL", (255,0,0)),
    3: ("VERDE", (0,255,0)),
    4: ("LARANJA", (0,165,255)),
}

while running:
    status, frame = cam.read()

    if not status or (cv.waitKey(1) & 0xff == ord('q')):
        running = False

    height, width, _ = frame.shape

    pt1, pt2 = ((width//2) - 100, (height//2) - 100), ((width//2) + 100, (height//2) + 100)
    region = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    b = cv.calcHist([region], [0], None, [256], (0, 256)).flatten()
    g = cv.calcHist([region], [1], None, [256], (0, 256)).flatten()
    r = cv.calcHist([region], [2], None, [256], (0, 256)).flatten()
    h = np.array(list(b) + list(g) +list(r))
    
    h = pca.transform([h])

    pred, color = dict_classes[knn.predict(h)[0]]

    cv.imshow("Detected", region)
    cv.rectangle(frame, pt1, pt2, color,thickness=3)
    cv.putText(frame, pred, (pt1[0], pt1[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv.LINE_AA)
    cv.imshow("Cam", frame)

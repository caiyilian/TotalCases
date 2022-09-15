import cv2
import numpy as np

index = 0
cap = cv2.VideoCapture(0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
imgBlack = np.zeros((height, width,3), dtype=np.uint8)
while True:
    success, img = cap.read()
    imgBlack[index:index + 1, 0:(-1)] = img[index:index + 1, 0:(-1)]
    imgBlack[index:(-1), 0:(-1)] = img[index:(-1), 0:(-1)]
    cv2.line(imgBlack, (0, index + 1), (width, index + 1), (255, 0, 0))
    index = index + 1
    cv2.imshow('img', imgBlack)
    cv2.waitKey(1)

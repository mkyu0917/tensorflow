import numpy as np
import cv2

def showimage():
    imgfile = 'images/7-1.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED) #이미지 읽기 플래그는 총 3가지가 있다.

    cv2.namedWindow('7-1', cv2.WINDOW_AUTOSIZE) #이미지 크기를 변경
    cv2.imshow('7-1', img)
    cv2.waitKey(0)
    cv2.destroyALLWindows()

showimage()
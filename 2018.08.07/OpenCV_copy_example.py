import numpy as np
import cv2

def showImage():
    imgfile = 'images/7-1.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR) #이미지 읽기 플래그는 총 3가지가 있다.
    cv2.imshow('7-1',img)

    k = cv2.waitKey(0) & 0xff

    if k == 27:
        cv2.destroyAllWindows()
    elif k==ord('c'):
        cv2.imwrite('images/7-1_copy.jpg', img)
        cv2.destroyAllWindows()


showImage()
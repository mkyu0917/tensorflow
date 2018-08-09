import numpy as np
import cv2

# def moment():
#     img = cv2.imread('D:\PycharmProject\data\model.jpg')
#     imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     ret, thr = cv2.threshold(imgray, 127, 255, 0)
#     _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     contour = contours[0]
#     mmt = cv2.moments(contour)
#
#     for key, val in mmt.items():
#         print('%s:\t%.5f'%(key, val))
#
#     cx = int(mmt['m10']/mmt['m00'])
#     cy = int(mmt['m01']/mmt['m00'])
#
#     print(cx, cy)
#
# moment()

def contour():
    img = cv2.imread('D:\PycharmProject\data\model.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray, 127, 255, 0)
    _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[164]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    cv2.drawContours(img, [cnt], 0, (255,255,0), 1)

    print('contour 면적: ', area)
    print('contour 길이: ', perimeter)

    cv2.imshow('contour', img)
    cv2.waitKey(0)
    cv2.destoryALLWindows()

contour()
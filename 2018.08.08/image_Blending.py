import cv2

def onMouse(x):
    pass
def imgBlending(imgfile1, imgfile2):
    img1=cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    cv2.namedWindow('ImgPane')
    cv2.createTrackbar('MIXING', 'ImgPane', 0, 100, onMouse)
    mix = cv2.getTrackbarPos('MIXING','ImgPane')
    while True:
        img=cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100,0)
        cv2.imshow('ImgPane',img)
        k=cv2.waitKey(1) & 0xFF
        mix = cv2.getTrackbarPos('MIXING', 'ImgPane')
cv2.destroyAllWindows()
imgBlending('D:/PycharmProject/data/suji.jpg','D:/PycharmProject/data/hallstatt.jpg')

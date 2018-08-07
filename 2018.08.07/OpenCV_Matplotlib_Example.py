import numpy as np
import cv2
import matplotlib.pyplot as plt

def showImage():
    imgfile = 'images/7-1.jpg'
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.title('7-1')
    plt.show()
showImage()
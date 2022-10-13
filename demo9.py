# 作者 : 杨航
# 开发时间 : 2022/10/13 15:45
# 9.单通道直方图
from matplotlib import pyplot as plt
import cv2
import numpy as np
img = cv2.imread('face.jpg')
cv2.imshow('before',img)
color = ('b','g','r')

for i,color in enumerate(color):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.title("photo")
    plt.xlabel('Bins')
    plt.ylabel('num of perlex')
    plt.plot(hist,color = color)
    plt.xlim([0,260])
plt.show()
cv2.waitKey(0)
cv2.destoryAllWindows()
# 作者 : 杨航
# 开发时间 : 2022/10/13 15:44
# 8.直方图绘制
from matplotlib import pyplot as plt
import cv2
import numpy as np
img = cv2.imread('face.jpg')

# 将图像转化为灰度图
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray,cmap=plt.cm.gray)
hist = cv2.calcHist([img],[0],None,[256],[0,256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel('# of Pixels')
plt.plot(hist)  # 折线图
plt.xlim([0,256]) # X轴
plt.show()
# 作者 : 杨航
# 开发时间 : 2022/10/13 15:44
# 7.将图片分为单通道，再将单通道转化为三通道
import cv2
import numpy as np
img = cv2.imread('face.jpg')
cv2.imshow('before',img)
cv2.waitKey(0)
b,g,r = cv2.split(img)

# 创建与img相同大小的零矩阵
zeros = np.zeros(img.shape[:2],dtype='uint8')
cv2.imshow('Bule',cv2.merge([b,zeros,zeros])) # 显示（b,0,0）图像
cv2.imshow('Green',cv2.merge([zeros,g,zeros]))# (0,g,0)
cv2.imshow('Red',cv2.merge([zeros,zeros,r]))  # (0,0,r)
cv2.waitKey(0)
cv2.destroyAllWindows()
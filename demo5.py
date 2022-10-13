# 作者 : 杨航
# 开发时间 : 2022/10/13 15:43
# 5.通道分离
import cv2
src = cv2.imread("face.jpg")
cv2.imshow("before",src)
# 调用通道分离
b,g,r = cv2.split(src)
# 三通道分别显示
cv2.imshow('blue',b)
cv2.imshow('green',g)
cv2.imshow('red',r)
cv2.waitKey(0)
cv2.destoryAllWindows()
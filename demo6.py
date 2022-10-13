# 作者 : 杨航
# 开发时间 : 2022/10/13 15:44
# 通道合并
import cv2
src = cv2.imread('face.jpg')
cv2.imshow('photo',src)
# 调用通道分离
b,g,r = cv2.split(src)
# 将Blue通道数值修改为0
g[:] = 0
# 合并修改后的通道
img_merge = cv2.merge([b,g,r])
cv2.imshow('after',img_merge)
cv2.waitKey(0)
cv2.destroyAllWindows()
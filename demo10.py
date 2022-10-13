# 作者 : 杨航
# 开发时间 : 2022/10/13 15:46
# 10.RGB转化为HSV
import cv2
image = cv2.imread('F:/PythonRepository/picture/4.png',1)
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
hsv = hsv[:, :, 2]
print(hsv.shape)
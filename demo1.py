# 作者 : 杨航
# 开发时间 : 2022/10/13 15:38

# 1.读入图像
##  cv2.IMREAD_COLOR彩色图像
##  cv2.IMREAD_GRAYSCALE灰度图像

# 导入opencv的python版本依赖库cv2
import cv2

# 使用opencv中imread函数读取图片
# 0代表灰度图形式打开，1代表彩色形式打开
img = cv2.imread("face.jpg",1)
print(img.shape)
# 高121 宽121  3通道： B G R
# 灰度图
img1 = cv2.imread('face.jpg',cv2.IMREAD_GRAYSCALE)
print(img1.shape)

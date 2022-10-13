# 作者 : 杨航
# 开发时间 : 2022/10/13 15:42
# 3.灰度转化
# 通道转化：三通道转为单通道灰度图
import cv2
# 以不改变的方式读取图片    cv2.IMREAD_UNCHANGED
img = cv2.imread('face.jpg',cv2.IMREAD_UNCHANGED)

# 查看打印图像的shape
shape = img.shape
print(shape)

#判断通道数是否为3通道或4通道
if shape[2] == 3 or shape[2] == 4:
    # 将彩色图转化为单通道图
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_image",img_gray) # 转化后的图
cv2.imshow("image",img) # 原图
cv2.waitKey(0)
cv2.destroyAllWindows()
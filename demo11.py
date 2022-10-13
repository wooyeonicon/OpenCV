# 作者 : 杨航
# 开发时间 : 2022/10/13 15:48
import cv2

# 11.模板匹配：一个模块与另一个模块的相似程度

import numpy as np
img1 = cv2.imread('face.jpg')
img2 = cv2.imread('eye.jpg')

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(img1,img2,cv2.TM_SQDIFF_NORMED) # 模板匹配
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)


# 12.图像金字塔
## 高斯金字塔
## 拉普拉斯金字塔

# 高斯金字塔：
    #  向下采样（缩小），往金字塔尖走，数据越来越少
        # 所有偶数行和列去除
    #  向上采样(扩大)
        # 扩大两倍，填充0
# 注意：先上采样，再下采样，结果和原始数据不一样，没有原始数据好

# 13.轮廓近似：将不规整的轮廓，近似成规整的轮廓
img = cv2.imread('star.jpg')
gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 二值处理
ret,threshold = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) # 二值处理
# 找轮廓
contours,hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = contours[3]

draw_img = img.copy()
# 画轮廓
res = cv2.drawContours(draw_img,[cnt],-1,(0,0,255),2)

cv2.imshow('result',res)
cv2.waitKey(0)

# 14.礼帽和黑帽
# 礼帽 = 原始输入 - 开运算结果
# 黑帽 = 闭运算结果 - 原始输入
img3 = cv2.imread('face.jpg')

tophat = cv2.morphologyEx(img3,cv2.MORPH_TOPHAT,kernel)

cv2.imshow('result',tophat)
cv2.waitKey(0)
cv2.destoryAllWindows()
blackhat = cv2.morphologyEx(img3,cv2.MORPH_BLACKHAT,kernel)

cv2.imshow('result',blackhat)
cv2.waitKey(0)
cv2.destoryAllWindows()

# 15.梯度运算
# 先对比两张图：
# 梯度运算 = 膨胀 - 腐蚀
pie = cv2.imread('face.jpg')
kernel = np.ones((7,7),np.uint8)
dilate = cv2.dilate(pie,kernel,iteration = 5)  # 膨胀
erosion = cv2.erode(pie,kernel,iteration = 5)  #腐蚀

res = np.hstack((dilate,erosion))

cv2.imshow('result',res)
cv2.waitKey(0)
cv2.destoryAllWindows()

# 16.图像融合
res = cv2.addWeighted(img1,0.4,img2,0.6,0)

# 17.边界填充
import cv2
img = cv2.imread('face.jpg')
# 填充的大小
top_size,bottom_size,left_size,right_size = (50,50,50,50)

# 不同的填充方法

# 复制法，复制边缘像素
replicate = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REPLICATE)

# 反射法，例如fedcba\abcdefgh\hgfedcba
reflect = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REFLECT)

# 反射法：以边缘像素为轴，gfedcb\abcdefgh\gfedcba
reflect101 = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REFLECT_101)

# 外包装法：cdefgh\abcdefgh\abcdefgh
wrap = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_WRAP)
# 常量法：常量填充
constant = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_CONSTANT,value=0)

# 18.读取视频
import cv2
vc = cv2.VideoCapture('class.mp4') # 读取视频

# 检查是否打开正确
if vc.isOpened():
    open,frame = vc.read()  # 一帧一帧读取
else:
    open = False

while open:
    ret,frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # 将一帧转化为灰度图
        cv2.imshow('result',gray)
        if cv2.waitKey(50) & 0xFF == 27:
            break
vc.release()
cv2.destoryAllWindows()
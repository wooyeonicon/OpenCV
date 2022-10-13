# 作者 : 杨航
# 开发时间 : 2022/10/13 15:42
# 4.RGB与BGR转化

# Opencv读取图像是以BGR形式读取，
# 但是有好多图像是RGB形式的。就需要RGB与BGR转化。

import cv2
img = cv2.imread('face.jpg',cv2.IMREAD_COLOR)
cv2.imshow("opencv_win",img)
# 用opencv自带的方法转
img_cv_method = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 作者 : 杨航
# 开发时间 : 2022/8/27 11:54
import cv2
import numpy as np
# 指定图片所在文件夹
filename = 'F:\\PythonRepository\\Deep-Learing\\pictures'
# 图像显示函数，全文唯一的自定义函数
def cv_show(name,img):
    cv2.imshow(name,img) # (自定义图像名,图像变量)
    cv2.waitKey(0) # 图像窗口不会自动关闭
    cv2.destroyAllWindows()  # 手动关闭窗口
# 读取模板图像
reference = cv2.imread(filename+'\\cardTemplate.png')  # 获取指定文件夹下的某张图片
cv_show('reference',reference) # 展示模板图
# 转换灰度图，颜色改变函数
ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
cv_show('gray',ref)
# 二值化处理，图像阈值函数，像素值超过127变成0，否则变成255
ret,thresh = cv2.threshold(ref,127,255,cv2.THRESH_BINARY_INV)
cv_show('threshold',thresh) # 返回值ret是阈值，thresh是二值化图像
# 轮廓检测。第1个参数是二值图。第2个参数检测最外层轮廓，第3个参数保留轮廓终点坐标
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 返回轮廓信息和轮廓层数
# 绘制轮廓
draw = reference.copy()  # 复制一份原图像作为画板，不能在原图上画，不然原图会改变
res = cv2.drawContours(draw, contours, -1, (0,0,255), 2)  #"-1"是指在画板上画出所有轮廓信息，红色，线宽为2
cv_show('res',res)
print(np.array(contours).shape)  # 显示有多少个轮廓  10个
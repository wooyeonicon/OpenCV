# 作者 : 杨航
# 开发时间 : 2022/10/13 15:41
# 2.图片展示和图片保存
import cv2
img = cv2.imread("face.jpg",1)
# 显示图像
# 参数：窗口名字，图片数据名
cv2.imshow('photo',img)
k = cv2.waitKey(0) # 等待0毫秒
if k == 27:   # 输入ESC键 退出
    cv2.destroyAllWindows()
elif k == ord('s'):  # 输入S键  保存图片并退出
    cv2.imwrite('man.jpg',img)
cv2.destroyAllWindows()
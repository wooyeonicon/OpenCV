# 作者 : 杨航
# 开发时间 : 2022/8/25 13:31
import cv2

def sort_contours(cnts,method='left-to-right'):
    reverse = False
    i = 0
    if method == 'right-to-left' or method == 'bottom-to -top':
        reverse = True
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    # 拿到每个轮廓左上角的坐标点，就可以对每个轮廓进行排序
    # 给每个轮廓画外接矩形
    boundingBoxes = [cv2.boundingRect[c] for c in cnts]# 用外接矩形找到x,y,w,h
    (cnts,boundingBoxes) = zip(*sorted(zip(cnts,boundingBoxes),
                                       key = lambda b:b[1][i],reverse=reverse))
    return cnts,boundingBoxes
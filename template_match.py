# 作者 : 杨航
# 开发时间 : 2022/8/25 13:12

import cv2
import numpy as np
import argparse
import myutils
from myutils import contours
# 设置参数
# 1.创建对象ArgumentParse()
ap = argparse.ArgumentParser()
# 2.调用add_argument()方法添加参数
ap.add_argument('-i','--image',required=True,help='path to input image')
ap.add_argument('-t','--template',required=True,help='path to template OCR-A image')
# 3.调用parse_args()解析添加的参数
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    '3':"American Express",
    '4':'Visa',
    '5':'MasterCard',
    '6':'Discover Card'
}

# 绘图展示
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destoryAllWindows()

# 读取一个模板图像
img = cv2.imread(args['template'])
cv_show('img',img)
# 灰度图
ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)
# 二值图像
ref = cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

# 轮廓检测
# 1.找到轮廓(外轮廓)
ref_,refCnts,hierarchy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# 2.画出轮廓，在原图上
cv2.drawContours(img,refCnts,-1,(0,0,255),3)
cv_show('img',img)
# 3.打印一共几个轮廓
print(np.array(refCnts).shape)
# 4.对轮廓进行排序
refCnts = myutils.sort_contours(refCnts,method='left-to-right')[0] # 排序，从左到右，从上到下

# 指定个字典
digits = {}
# i:轮廓索引   c:轮廓
for (i,c) in enumerate(refCnts):
    # 外接矩形
    (x,y,w,h) = cv2.boundingRect(c)
    roi = ref[y:y+h,x:x+w]  # 扣出这个矩形（拿到这个外接矩形）
    roi = cv2.resize(roi,(57,88)) # 调整合适的大小
    digits[i] = roi # 将索引与外接矩形通过键值对存起来


# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# 读取输入图像，预处理
image = cv2.imread(args['image'])
cv_show('image',image)
image = myutils.resize(image,width=300)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

# 礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv_show('tophat',tophat)

# ksize=-1相当于3*3
gradX = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)

gradX = np.absolute(gradX)  # 绝对值
(minVal,maxVal) = (np.min(gradX),np.max(gradX))
gradX = (255*((gradX-minVal)/(maxVal-minVal)))
gradX = gradX.astype('uint8')

print(np.array(gradX).shape)
cv_show('gradX',gradX)

# 使用闭操作，先膨胀再腐蚀（先连在一起，腐蚀不掉），得到每一个组的数字（其中包含4个数字）
gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
cv_show('gradX',gradX)

# 自动找寻合适的阈值(二值化操作)
# THRESH_OTSU 两种结果，双峰，适合使用
thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

# 再闭操作(把中间小空隙填补)
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
cv_show('thresh',thresh)

# 计算轮廓
thresh_,threshCnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# 在原图上画出轮廓
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('img',cur_img)


locs = []
# 遍历轮廓
for (i,c) in enumerate(cnts):
    # 计算外接矩形
    (x,y,w,h) = cv2.boundingRect(c) # 外接矩形
    ar = w/float(h)  # 计算比例

    # 选择合适的区域，根据实际任务，这里基本都是四个数字一组
    if ar > 2.5 and ar < 4.0:
        if(w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x,y,w,h))

# 符合的大轮廓进行排序（从左到右）
locs = sorted(locs,key=lambda x:x[0])
output = []

# 通过遍历么一个轮廓中的数字
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    groupOutput = []

    # 将轮廓稍微扩大点
    group = grad[gY -5:gY+gH+5,gX-5:gX+gW+5]
    cv_show('group',group)

    #预处理(大轮廓中。轮廓检测，找到每一个小轮廓)
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # 二值化
    cv_show('group',group)

    # 计算每一组的轮廓
    group_,digitCnts,hierarchy = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts,method='left-to-right')[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的大小
        (x,y,w,h) = cv2.boundingRect(c)
        roi = group[y:y+h,x:x+w]
        roi = cv2.resize(roi,(57,88))
        cv_show('roi',roi)

        # 计算匹配得分
        scores = []
        # 在模板中计算每一个得分
        for (digit,digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
            (_,score,_,_) = cv2.minMaxLoc(result)
            score.append(score)
        # 得分最合适的数字
        groupOutput.append(str(np.argmax(scores)))
    # 画出来
    cv2.rectangle(image,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX,gY-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)

    # 得到结果
    output.extend(groupOutput)
# 打印结果
print('Credict Card Type:{}'.format(FIRST_NUMBER[output[0]]))
print('Credict Card # : {}'.format(''.join(output)))
cv2.imshow('image',image)
cv2.waitKey(0)


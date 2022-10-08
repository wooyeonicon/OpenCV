# 作者 : 杨航 stackoverflow
# 开发时间 : 2022/8/25 21:29
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 指定图片所在的文件夹
filename = 'F:\\PythonRepository\\Deep-Learing\\pictures'

# 图片显示函数
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destoryAllWindows()

#########################################
# 1.模板处理
# 1.1 模板预处理
# 读取模板图像
# img = cv2.imread(filename+'\\cardTemplate.JPG')
img = cv2.imread(filename+'\\cardTemplate.JPG')
#cv_show('img',img)

# 转换为灰度图
ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv_show('ref',ref)


# 二值化处理。图像阈值函数，像素值超过127变成0，否则变成255
ret,thresh = cv2.threshold(ref,127,255,cv2.THRESH_BINARY_INV)
# cv_show('thresh',thresh)  # ret是阈值

# 轮廓检测：画出外轮廓（第一个参数：二值图。第二个参数：检测最外层轮廓。第三个参数：保留轮廓终点坐标）
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # 返回轮廓信息和轮廓层数
#绘制轮廓(在原图的复制图上)
res = cv2.drawContours(img.copy(),contours,-1,(0,0,255),2)
#cv_show('res',res)
print(np.array(contours).shape)  # 显示有多少个轮廓，10个

# 1.2模板排序（轮廓检测得到了10个数字的轮廓，但是这10个轮廓不一定是按顺序排列的）
# 求每一个轮廓的外接矩形，根据返回的左上坐标点，就能判断出轮廓的位置，再排序
# boxing中存放每次计算轮廓外接矩形得到的x、y、w、h，它的shape为(10,4)。cnt存放每一个轮廓
boxing = [np.array(cv2.boundingRect(cnt)) for cnt in contours]
contours = np.array(contours)
# 都变成数组类型，为了下面冒泡排序能相互交换值。上面默认返回的是元组类型，它只能读不能写
# 把x坐标最小轮廓的排在第一个
for i in range(9):  #冒泡排序
    for j in range(i+1,10):
        if boxing[i][0]>boxing[j][0]:  #把x坐标大的值放到后面
        # 给boxing中的值换位置
            boxing[i],boxing[j] = boxing[j],boxing[i]
        # 给轮廓信息换位置
            contours[i],contours[j] = contours[j],contours[i]

# 1.3 模板数字区域对应具体数字
# 指定个字典
digits = {}
# i:轮廓索引   c:轮廓
for (i,c) in enumerate(contours):
    # 外接矩形
    (x,y,w,h) = boxing[i] # boxing中存放的是每个轮廓的信息
    roi = ref[y:y+h,x:x+w]  # 扣出这个矩形（拿到这个外接矩形）(不至于太挤)
    roi = cv2.resize(roi,(57,88)) # 调整合适的大小
    digits[i] = roi # 将索引与外接矩形通过键值对存起来
    plt.subplot(2, 5, i + 1)
    plt.imshow(roi, 'gray'), plt.xticks([]), plt.yticks([])  # 不显示xy轴刻度
#######################################################################

##########################################################################
# 2.Card处理
# 2.1 读取
img1 = cv2.imread(filename+'\\card.jpg')
#cv_show('ref1',img1)
ref1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#cv_show('ref1',ref1)


# 2.2形态学：礼帽（突出更明亮的区域）
# 定义卷积核，MORPH_RECT矩形，size为9*3
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3)) # 长*宽
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

tophat = cv2.morphologyEx(ref1,cv2.MORPH_TOPHAT,rectKernel)
#cv_show('tophat',tophat)

####################################################################################
# 3.边缘检测
# 3.1:边缘检测,使用canny方法cv2.Canny()，也可以使用sobel方法。
img_canny = cv2.Canny(tophat,80,200)  # 自定义最小和最大阈值
#cv_show('canny',img_canny)


# 3.2形态学：闭操作
# 先膨胀后腐蚀。用于填补内部黑色小空洞，填充图像。
# 使用卷积核9*3，做三次迭代
#img_close = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, rectKernel,iterations=12)
img_close = cv2.morphologyEx(img_canny,cv2.MORPH_CLOSE,rectKernel,iterations = 2)
#cv_show('close',img_close)


# 3.3轮廓检测
threshCnts,hierarchy = cv2.findContours(img_close.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# 3.4在原图上画出轮廓

res = cv2.drawContours(img1.copy(),threshCnts,-1,(0,255,0),1)
#cv_show('cur_img',res)

# 3.4轮廓筛选
loc = [] # 存放图片排序后的轮廓要素
mess = [] # 每图片排序后的轮廓信息
for (i,c) in enumerate(threshCnts): #返回下标和对应值
    # 每一个轮廓的外接矩形要素
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)  # 计算长宽比
    # 选择合适的长宽比
    if ar>2.5 and ar<4:
        if (w>40 and w<70) and (h>10 and h<30):
            # 符合要求的留下
            loc.append((x,y,w,h))
            mess.append(c)
# 3.5 轮廓排序
# 将符合的轮廓从左到右排序
# 把x坐标最小轮廓的排在第一个
for i in range(len(loc)-1):  #冒泡排序
    for j in range(i+1,len(loc)):
        if loc[i][0]>loc[j][0]:  #把x坐标大的值放到后面
            # 交换轮廓要素信息
            loc[i],loc[j] = loc[j],loc[i]
            # 交换对应的轮廓信息
            mess[i],mess[j] = mess[j],mess[i]

# 3.6 轮廓内数字提取
output = []  # 保存最终数字识别结果
for (i,(x,y,w,h)) in enumerate(loc):  # loc中存放的是每个组合的xywh
    groupOutput = [] # 存放取出来的数字组合
    group = ref1[y - 5:y + h + 5, x - 5:x + w + 5]  # 每个组合的坐标范围是[x:x+w][y:y+h]，加减5是为了给周围留点空间

    # 每次去除的轮廓二值化
    ret,group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv_show('group',group)

    # 每个数字的小轮廓检测，只检测最外层
    contours, hierarchy = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 返回轮廓信息和轮廓层数
    res = cv2.drawContours(img1.copy(), contours, -1, (0, 255, 0), 1)
    #cv_show('res',res)

    # 对轮廓排序，boxing 中存放每次计算轮廓外界接矩形得到的x,y,w,h
    boxing = [np.array(cv2.boundingRect(cnt)) for cnt in contours]
    contours = np.array(contours)  # 都变成数组类型，下面冒泡排序能相互交换值，元组类型

    # 把x坐标最小轮廓的排在第一个
    for i in range(3):
        for j in range(i+1,4):
            if boxing[i][0]>boxing[j][0]: # 把x坐标大的值放到后面
                # 给boxing中的值交换位置
                boxing[i],boxing[j] = boxing[j],boxing[i]
                # 给轮廓信息换位置
                contours[i],contours[j] = contours[j],contours[i]

    # 给排序后的轮廓分别计算每一个数字组合中的每一个数字

    for c in contours: # c代表每一个小数字的轮廓
        (gx,gy,gw,gh) = cv2.boundingRect(c) # 计算每个数字的轮廓的x,y,w,h
        roi = group[gy:gy+gh,gx:gx+gw]  # 在数字组合中扣除每一个数字区域
        roi = cv2.resize(roi,(57,88))  # 大小和最开始resize的模板大小一样
        #cv_show('roi',roi)  # 扣出了所有的数字

        # 开始匹配
        score = [] # 定义模块匹配度得分变量

        # 从模块中逐一取出数字和刚取出的roi比较
        for (dic_key, dic_value) in digits.items():  # items()函数从我们最开始定义的模板字典中取出索引和值
            # 模板匹配，计算归一化相关系数cv2.TM_CCOEFF_NORMED，计算结果越接近1，越相关
            res = cv2.matchTemplate(roi, dic_value, cv2.TM_CCOEFF_NORMED)
            # 返回最值及最值位置，在这里我们需要的是最小值的得分，不同的匹配度计算方法的选择不同
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            score.append(max_val)
        # 当roi与模板中的10个数比较完以后，score中保存的是roi和每一个数字匹配后的结果
        # score中值最小的位置，就是roi对应的数字
        print('===========')
        score = np.abs(score)  # 有负数出现，统一成正数，相关系数都变成正数
        best_index = np.argmax(score)  # score最大值的下标，匹配度最高
        best_value = str(best_index)  # 下标就是对应的数字，在字典中，key是0对应的是值为0的图片
        groupOutput.append(best_value)  # 将对应的数字保存
    # 打印识别结果
    print("结果：",groupOutput)

    # 把识别处理的数字在原图上画出来，指定矩形框的左上角坐标和右下角坐标
    cv2.rectangle(img1,(x-5,y-5),(x+w+5,y+h+5),(0,0,255),1)
    # 在矩形框上绘图
    cv2.putText(img1,''.join(groupOutput),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
    # 将得到的数字结果保存在一起
    output.append(groupOutput)

cv_show('img',img1)
print('数字为：',output)

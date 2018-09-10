#加载需要用到的库
import tensorflow as tf
import numpy as np
import cv2
import re
import glob
import skimage
import matplotlib.pyplot as plt
from PIL import Image
#初始化位置
pos=[]


# 画出轮廓
# https://blog.csdn.net/sinat_36458870/article/details/78825571  参考网站

def Draw_Bound(imageName):
    image = cv2.imread(imageName)
    a, b, c = image.shape
    # 转换灰度并去噪声
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # 提取图像的梯度
    # 从梯度y减去梯度x
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # cv2.imshow('imag',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 使用低通滤泼器模糊图像二值化
    # 可以平滑并替代那些强度变化明显的区域
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    # (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    # 填充内部空间
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    (_, cnts, _) = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    # 找出轮廓
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    # cv2.imshow("Image", image)
    # cv2.imwrite("contoursImage2.jpg", image)
    #  cv2.waitKey(0)
    # 其实，box里保存的是绿色矩形区域四个顶点的坐标
    # 找出四个顶点的x，y坐标的最大最小值。新图像的高=maxY-minY，宽=maxX-minX
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    hight_prob = int(hight / 20)
    width_prob = int(width / 20)
    cropImg = image[y1 + hight_prob * 2:y1 + hight - hight_prob * 2, x1 + width_prob * 2:x1 + width - width_prob * 2]
    pos.append(x1 + width_prob * 2)
    pos.append(y1 + hight_prob * 2)
    pos.append((x1 + width - width_prob * 2))
    pos.append((y1 + hight - hight_prob * 2))
    # cv2.imshow('imag',cropImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropImg.copy()
    # print(cropImg[int(cropImg.shape[1]/2)][int(cropImg.shape[0]/2)])


#将脸P上处理好的图片上
def facing(img,face,index1,index2):
    img[index1[0]:index2[0],index1[1]:index2[1]]=face[0:face.shape[0],0:face.shape[1]]


#找到要P上的脸部的位置，魔法随缘参数
def find_face_pos(img):
    #将脸缩放至原图1/4大小
    high_prob=1/4
    face=cv2.imread('pkq_face.jpg')
    face=cv2.resize(face,(int(img.shape[0]*high_prob),int(img.shape[1]*high_prob)))
    #逐个像素点按照设定好的RGB值改变
    #从皮卡丘原图提取出来的RGB值，用于染色
    for i in range(img.shape[0]-face.shape[0]):
        for j in range(img.shape[1]-face.shape[1]):
            if ((img[i,j,0]==74)and(img[i,j,1]==203)and(img[i,j,2]==237)and(img[i,j+face.shape[1],0]==74)and(img[i,j+face.shape[1],1]==203)and(img[i,j+face.shape[1],2]==237)and (img[i+face.shape[0],j,0]==74)and(img[i+face.shape[0],j,1]==203)and(img[i+face.shape[0],j,2]==237) and (img[i+face.shape[0],j+face.shape[1],0]==74)and(img[i+face.shape[0],j+face.shape[1],1]==203)and(img[i+face.shape[0],j+face.shape[1],2]==237)):
                facing(img,face,[i,j],[i+face.shape[0],j+face.shape[1]])
                return img.copy()


def fill_color_demo(image):
    """
    泛洪法上色：会改变图像
    """
    # 复制图片
    copyImg = image.copy()
    # 获取图片的高和宽
    h, w = image.shape[:2]

    # 创建一个h+2,w+2的遮罩层，
    # 这里需要注意，OpenCV的默认规定，
    # 遮罩层的shape必须是h+2，w+2并且必须是单通道8位，具体原因我也不是很清楚。
    mask = np.zeros([h + 2, w + 2], np.uint8)

    # 这里执行漫水填充，参数代表：
    # copyImg：要填充的图片
    # mask：遮罩层
    # (30,30)：开始填充的位置（开始的种子点）
    # (0,255,255)：填充的值，这里填充成黄色
    # (100,100,100)：开始的种子点与整个图像的像素值的最大的负差值
    # (50,50,50)：开始的种子点与整个图像的像素值的最大的正差值
    # cv.FLOODFILL_FIXED_RANGE：处理图像的方法，一般处理彩色图象用这个方法
    cv2.floodFill(copyImg, mask, (100, 100), (74, 203, 237), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    return copyImg

#将P好的轮廓图 放回原图里
def put(IMG,face,box):
    #img = cv2.imread("plane.jpg")
    #cv2.imshow("OpenCV",img)
    #print(box,IMG.shape,face.shape)
    image = Image.fromarray(cv2.cvtColor(IMG,cv2.COLOR_BGR2RGB))
    face = Image.fromarray(cv2.cvtColor(face,cv2.COLOR_BGR2RGB))
    image.paste(face,box)
    return image.copy()

"""
---进行图像批处理
百度爬了一部分的图，进行批处理
基本上能够正确P上

for picture in glob.glob(r'Pic/*.jpeg'):
    #crop_IMG=Draw_Bound('anot_th.jpeg')
    crop_IMG=Draw_Bound(picture)
    filled_img=fill_color_demo(crop_IMG.copy())
    faced_img=find_face_pos(filled_img.copy())
    origin=cv2.imread(picture)
    pasted_img=put(origin,faced_img,pos)
    #正则表达式将修改后图片存到另一个文件夹Pic->done_pic
    picture=re.sub(r'Pic','done_pic',picture)
    pasted_img.save(picture)
    pos=[]
"""

if __name__=="__main__":
    picture=input("输入图片名字: ")
    """
    处理过程
    """
    #======  因为需要用到皮卡丘脸的图来P上，所以要将pkq_face.jpg放到同一个文件夹里  =====

    crop_IMG = Draw_Bound(picture)
    filled_img = fill_color_demo(crop_IMG.copy())
    faced_img = find_face_pos(filled_img.copy())
    origin = cv2.imread(picture)
    pasted_img = put(origin, faced_img, pos)
    #cv2.imshow("PIKACHU",pasted_img)
    Image._show(pasted_img)
    #初始化位置变量
    pos=[]

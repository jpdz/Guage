import cv2
import numpy as np
from PIL import Image
import math

''' 找到圆心和半径'''
def get_info_circle(bitwise):
    circles= cv2.HoughCircles(bitwise,cv2.HOUGH_GRADIENT,1,50,param1=80,param2=30,minRadius=15,maxRadius=200)
    for circle in circles[0]:
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])
        print(x,y,r)
        ninfo = [r, x, y]
        return ninfo

'''圆环转直线，圆环半径自设 '''
def get_huan_by_circle(img,circle_center,radius,radius_width):
    black_img = np.zeros((radius_width,int(2*radius*math.pi)),dtype='uint8')
    for row in range(0,black_img.shape[0]):
        for col in range(0,black_img.shape[1]):
            theta = math.pi*2/black_img.shape[1]*(col+1)#+origin_theta
            rho = radius-row-1
            p_x = int(circle_center[0] - rho*math.sin(theta)+0.5)-1
            p_y = int(circle_center[1] + rho*math.cos(theta)+0.5)-1
            
            black_img[row,col] = img[p_y,p_x]
    return black_img

''' 水平投影和垂直投影'''
def getHProjection(image):
    hProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    h,w=image.shape
    #长度与图像高度一致的数组
    h_ = [0]*h
    #循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 255:
                h_[y]+=1
    #绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y,x] = 255
    return h_
 
def getVProjection(image):
    vProjection = np.zeros(image.shape,np.uint8);
    #图像高与宽
    h,w = image.shape
    #长度与图像宽度一致的数组
    w_ = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 255:
                w_[x]+=1
    #绘制垂直平投影图像
    for x in range(w):
        for y in range(h-w_[x],h):
            vProjection[y,x] = 255
    cv2.imwrite('0_verticle.jpg', vProjection)
    return w_

''' 数字拉伸成水平线上时，截取垂直投影数字'''
def cut(im):
    W = getVProjection(im)
    Wstart = 0
    Wend = 0
    inBlock = False
    for i in range(im.shape[1]):
        if not inBlock and W[i] !=0:
            inBlock = True
            Wstart = i
        elif W[i]==0 and inBlock:
            Wend = i
            inBlock = False
            res = im[:,Wstart:Wend+1]
            print(Wstart,Wend)
            cv2.imwrite("%d_res.jpg"%i,res)


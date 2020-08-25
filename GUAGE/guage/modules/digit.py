import datetime
from random import sample
import cv2
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from sympy import *
import math
from utils import *


'''预处理 '''
class METHOD():
    def get_max_point(self, cnt):
        lmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        tmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bmost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        pmost = [lmost, rmost, tmost, bmost]
        return pmost

    def distance(self, pmost, centerpoint):
        cx, cy = centerpoint
        distantion = []
        for point in pmost:
            dx, dy = point
            distantion.append((cx - dx) ** 2 + (cy - dy) ** 2)
        index_of_max = distantion.index((max(distantion)))
        return index_of_max

    def ds_ofpoint(self, a, b):
        x1, y1 = a
        x2, y2 = b
        distances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        return distances

    def findline(self, cp, lines):
        x, y = cp
        cntareas = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            aa = sqrt(min((x1 - x) ** 2 + (y1 - x) ** 2, (x2 - x) ** 2 + (y2 - x) ** 2))
            if (aa < 50):
                cntareas.append(line)
        return cntareas

'''根据筛选的流通域截取数字,以及数字的中心点坐标 '''
def cut_image(digitset,img):
    midpoint = []
    for i in range(len(digitset)):
        rect = cv2.minAreaRect(digitset[i])
        midpoint.append(rect)
    lst = sorted(midpoint, key=lambda x: x[0][1])
    for i,rect in enumerate(lst):
        a,b,c = rect
        ### 数字的中心点坐标
        print(a)
        top_left = [a[0]-b[0]/2,a[1]-b[1]/2]
        bottom_right = [a[0]+b[0]/2,a[1]+b[1]/2]
        top_left = [int(x) for x in top_left]
        bottom_right = [int(x) for x in bottom_right]
        new = img[top_left[1]-10:bottom_right[1] + 10, top_left[0]-10:bottom_right[0] + 10]
        cv2.imwrite("result2/%d.jpg"%i,new)

def linecontours(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cp_info = get_info_circle(gray)
    r_1, c_x, c_y = cp_info
    binary = cv2.adaptiveThreshold(~gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ##存储数据
    digitset = []
    ca = (c_x, c_y)
    incircle = [r_1 * 0.5, r_1 * 0.7, r_1*0.9]
    for xx in contours:
        rect = cv2.minAreaRect(xx)
        ## 最小外接矩，c是偏移角度
        a, b, c = rect
        w, h = b
        w = int(w)
        h = int(h)
        ## 满足条件:“长宽比例”，“面积”等'''
        if h == 0 or w == 0:
            pass
        else:
            dis = METHOD.ds_ofpoint(self=0, a=ca, b=a)
            if(abs(c+90)<1 or -7<c<=0 and dis<r_1):
                ###digitset.append(xx)
                ###可以加一个判断条件，到圆心的距离在incircle中找 
                if (incircle[2] > dis >incircle[1]):
                    digitset.append(xx)            
    cv2.drawContours(img, digitset, -1, (255, 90, 60), 2)
    cv2.imshow("1",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cut_image(digitset,binary) 


if __name__ == "__main__":
    linecontours("test.jpg")




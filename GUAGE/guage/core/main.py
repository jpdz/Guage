import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from math import cos, pi, sin
from help import *

if __name__ == '__main__':
    for x in range(1,32):
        #获取测试图像
        img_s = cv2.imread('../images/%d.jpg'%x)
        img=cv2.cvtColor(img_s,cv2.COLOR_BGR2GRAY)
        template = cv2.imread('template1.png')
        template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        #匹配并返回矩形坐标
        top_left,bottom_right=get_match_rect(template,img,method)
        c_x,c_y=get_center_point(top_left,bottom_right)
        #绘制矩形
        cv2.rectangle(img_s, top_left, bottom_right, 255, 2)
        new = img_s[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        template = cv2.imread('template.png')
        top_left, bottom_right = get_match_rect(template, new, method=method)
        new_ = new[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        # 二值化图像
        if (args.p==["1"]):
            img=v2_by_k_means(new_)
            #img = v2_by_method(new_)
            cv2.imwrite("test.jpg",img)
            rad=get_pointer_rad(img)
            print(rad)
            print('对应刻度',get_rad_val(rad[1]))
            print('#################################  next image #####################')
        ### 第二种图片
        else:
            img = cv2.imread('0.jpg')
            img=v2_by_k_means(img)
            rad=get_pointer_rad(img)
            print(rad)
            print('对应刻度',get_rad_val(rad[1]))
            print('#################################  next image #####################')

cv2.destroyAllWindows()



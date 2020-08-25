import math
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from math import cos, pi, sin

p = argparse.ArgumentParser()
p.add_argument('-p',help='Optional parameters',action = 'append',default = ["1"])
args = p.parse_args()

'''获取模板匹配的矩形的左上角和右下角的坐标'''
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
method = cv2.TM_CCOEFF

def get_match_rect(template,img,method):
    w, h = template.shape[1],template.shape[0]
    res = cv2.matchTemplate(img, template, method)
    mn_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的方法，对结果的解释不同
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left,bottom_right

'''传入左上角和右下角坐标，获取中心点'''
def get_center_point(top_left,bottom_right):
    c_x, c_y = ((np.array(top_left) + np.array(bottom_right)) / 2).astype(np.int)
    return c_x,c_y

'''获取中心圆形区域的色值集'''
def get_circle_field_color(img,center,r,thickness):    
    temp=img.copy().astype(np.int)
    cv2.circle(temp,center,r,-100,thickness=thickness)
    return img[temp == -100]

'''二值化方法1'''
def v2_by_method(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    return binary

'''二值化方法2：通过中心圆的颜色集合'''
def v2_by_center_circle(img,colors):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a = img[i, j]
            if a in colors:
                img[i, j] = 0
            else:
                img[i, j] = 255

''' 二值化方法3：使用k-means二值化'''
def v2_by_k_means(img):
    original_img = np.array(img, dtype=np.float64)
    src = original_img.copy()
    delta_y = int(original_img.shape[0] * (0.4))
    delta_x = int(original_img.shape[1] * (0.4))
    original_img = original_img[delta_y:-delta_y, delta_x:-delta_x]
    h, w, d = src.shape
    print(w, h, d)
    dts = min([w, h])
    print(dts)
    r2 = (dts / 2) ** 2
    c_x, c_y = w / 2, h / 2
    a: np.ndarray = original_img[:, :, 0:3].astype(np.uint8)
    # 获取尺寸(宽度、长度、深度)
    height, width = original_img.shape[0], original_img.shape[1]
    depth = 3
    print(depth)
    image_flattened = np.reshape(original_img, (width * height, depth))
    '''
    用K-Means算法在随机中选择1000个颜色样本中建立64个类。
    每个类都可能是压缩调色板中的一种颜色。
    '''
    image_array_sample = shuffle(image_flattened, random_state=0)
    estimator = KMeans(n_clusters=2, random_state=0)
    estimator.fit(image_array_sample)
    ###我们为原始图片的每个像素进行类的分配
    src_shape = src.shape
    new_img_flattened = np.reshape(src, (src_shape[0] * src_shape[1], depth))
    cluster_assignments = estimator.predict(new_img_flattened)
    ### 我们建立通过压缩调色板和类分配结果创建压缩后的图片
    compressed_palette = estimator.cluster_centers_
    print(compressed_palette)
    a = np.apply_along_axis(func1d=lambda x: np.uint8(compressed_palette[x]), arr=cluster_assignments, axis=0)
    img = a.reshape(src_shape[0], src_shape[1], depth)
    print(compressed_palette[0, 0])
    threshold = (compressed_palette[0, 0] + compressed_palette[1, 0]) / 2
    img[img[:, :, 0] > threshold] = 255
    img[img[:, :, 0] < threshold] = 0
    for x in range(w):
        for y in range(h):
            distance = ((x - c_x) ** 2 + (y - c_y) ** 2)
            if distance > r2:
                pass
                img[y, x] = (255, 255, 255)
    return img

'''获取角度'''
def get_pointer_rad(img):
    
    shape = img.shape
    c_y, c_x, depth = int(shape[0] / 2), int(shape[1] / 2), shape[2]
    x1=c_x+c_x*0.8
    src = img.copy()
    freq_list = []
    for i in range(361):
        x = (x1 - c_x) * cos(i * pi / 180) + c_x
        y = (x1 - c_x) * sin(i * pi / 180) + c_y
        temp = src.copy()
        cv2.line(temp, (c_x, c_y), (int(x), int(y)), (0, 0, 255), thickness=3)
        t1 = img.copy()
        t1[temp[:, :, 2] == 255] = 255
        c = img[temp[:, :, 2] == 255]
        points = c[c == 0]
        freq_list.append((len(points), i))
        cv2.imshow('d1', t1)
        cv2.waitKey(1)
    print('当前角度：',max(freq_list, key=lambda x: x[0]),'度')
    cv2.destroyAllWindows()
    return max(freq_list, key=lambda x: x[0])

'''第一种图片 or 第二种图片 '''
if (args.p==["1"]):
    center=[121 , 116]

    a={20:( 52,189 ) ,30:( 34,168 ) ,40: ( 24,144 ),50:( 22,103 ) ,
       60:( 40,60 ) ,70:( 90,25 ) ,80:( 166,31 ) ,90:( 218,90 ) ,100:(193,186)}
    count=0
    result={}
    for k ,v in a.items():
        r=math.acos((v[0]-center[0])/((v[0]-center[0])**2 + (v[1]-center[1])**2)**0.5)
        r=r*180/math.pi
        a[k]=r
        if count >= 4 and k != 100:
            r=360-r
            # print(k, r)
        result[k]=r
        count+=1
    d=360-result[90]+result[100]
    d1=360-result[90]
    t=90+(100-90)*(d1/d)
    result[t]=0
    result_list=result.items()
    lst=sorted(result_list,key=lambda x:x[1])
    print(lst)

else:
    center = [122,122]
    a={25:(120,56),5:(49,121),45:(188,122),-5:(64,170),55:(165,174)}
    result={}
    for k ,v in a.items():
        r=math.acos((v[0]-center[0])/((v[0]-center[0])**2 + (v[1]-center[1])**2)**0.5)
        r=r*180/math.pi
        if(k==25):
            r = 360-r
        result[k] = r
    result_list=result.items()
    lst=sorted(result_list,key=lambda x:x[1])
    print(lst)

''' 下一个刻度数字 '''
def get_next(c):
    l=len(lst)
    n=0
    for i in range(len(lst)):
        if lst[i][0]==c:
            n=i+1
            if n==l:
                n=0
            break
    return lst[n]

''' 指针指向结果 '''
def get_rad_val(rad):
    left=None
    for k, v in lst:
        # print(k,v)
        if rad > v :
            left = k
    left_rad=result[left]
    gap=rad-left_rad
    right=get_next(left)
    t=left+10*abs(gap/(right[1] - left_rad))
    print(t)
    return t

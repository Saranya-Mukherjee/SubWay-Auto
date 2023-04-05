import numpy
import numpy as np
import cv2
from mss import mss
from PIL import Image
import matplotlib as plt

bounding_box = {'top': 35, 'left': 1920, 'width': 1600, 'height': 900}
threshold=0.6

sct = mss()

def draw(location, search, size):
    for pt in zip(*location[::-1]):
        cv2.rectangle(search,pt,(pt[0]+size[0],pt[1]+size[1]),(0,0,225),2)
    return search

def Train1(search):
    tem=cv2.cvtColor(cv2.imread("D:\Subway AUTO\Train 1.jpg"), cv2.COLOR_BGR2GRAY)
    w,h=tem.shape[::-1]
    res=cv2.matchTemplate(search,tem,cv2.TM_CCOEFF_NORMED)
    loc=numpy.where(res>=threshold)
    s=(w,h)
    search = draw(loc, search, s)
    return search

def Train2(search):
    tem=cv2.cvtColor(cv2.imread("D:\Subway AUTO\Train 2.jpg"), cv2.COLOR_BGR2GRAY)
    w,h=tem.shape[::-1]
    res=cv2.matchTemplate(search,tem,cv2.TM_CCOEFF_NORMED)
    loc=numpy.where(res>=threshold)
    s=(w,h)
    search = draw(loc, search, s)
    return search

def Train3(search):
    tem=cv2.cvtColor(cv2.imread("D:\Subway AUTO\Train 2.jpg"), cv2.COLOR_BGR2GRAY)
    w,h=tem.shape[::-1]
    res=cv2.matchTemplate(search,tem,cv2.TM_CCOEFF_NORMED)
    loc=numpy.where(res>=threshold)
    s=(w,h)
    search = draw(loc, search, s)
    return search

def Jump1(search):
    tem=cv2.cvtColor(cv2.imread("D:\Subway AUTO\Jump 1.png"), cv2.COLOR_BGR2GRAY)
    w,h=tem.shape[::-1]
    res=cv2.matchTemplate(search,tem,cv2.TM_CCOEFF_NORMED)
    loc=numpy.where(res>=threshold)
    s=(w,h)
    search=draw(loc,search, s)
    return search

def Jump_global(search):
    tem=cv2.cvtColor(cv2.imread("D:\Subway AUTO\Jump 1.png"),cv2.COLOR_BGR2GRAY)
    w,h=tem.shape[::-1]
    s=(w,h)
    res=cv2.matchTemplate(search,tem,cv2.TM_CCOEFF_THRESHOLD)
    threshold_tem=threshold+float(input("Variable threshold: "))
    loc=numpy.where(res>=threshold)
    ret, th1 = cv2.threshold(tem, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(tem, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(threshold_tem, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 11, 2)
    tem_final=cv2.add(th1+th2)
    tem_final=cv2.add(tem_final+th3)
    tem_overlay=cv2.add(tem_final+threshold_tem)
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(res, 'gray')
    plt.show()

def Jump2(search):
    tem=cv2.cvtColor(cv2.imread("D:\Subway AUTO\Jump 1.png"), cv2.COLOR_BGR2GRAY)
    w,h=tem.shape[::-1]
    res=cv2.matchTemplate(search,tem,cv2.TM_CCOEFF_NORMED)
    loc=numpy.where(res>=threshold)
    s=(w,h)
    search=draw(loc,search, s)
    return search

def loop():
    pass

while True:
    sct_img = sct.grab(bounding_box)
    frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)
    frame=Train1(frame)
    frame=Train2(frame)
    frame=Jump1(frame)
    cv2.imshow('screen', frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 21:59:41 2021

@author: Lenovo
"""


import cv2 
import numpy as np 

pic_name = r'./play_ground.png'

def clahe(img, clip_limit=2.0, grid_size=(8,8)): 
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size) 
    return clahe.apply(img) 

src = cv2.imread(pic_name) 


# HSV thresholding to get rid of as much background as possible 
hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV) 
lower_blue = np.array([0, 0, 120]) 
upper_blue = np.array([180, 38, 255]) 
mask = cv2.inRange(hsv, lower_blue, upper_blue) 
result = cv2.bitwise_and(src, src, mask=mask) 
b, g, r = cv2.split(result) 
g = clahe(g, 5, (3, 3)) 

# Adaptive Thresholding to isolate the bed 
img_blur = cv2.blur(g, (9, 9)) 
img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
           cv2.THRESH_BINARY, 51, 2) 

contours, hierarchy = cv2.findContours(img_th, 
              cv2.RETR_CCOMP, 
              cv2.CHAIN_APPROX_SIMPLE) 

# Filter the rectangle by choosing only the big ones 
# and choose the brightest rectangle as the bed 

contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
max_brightness = 0 
canvas = src.copy() 


for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        cv2.drawContours(src, [approx], -1, (0,255,0), 3)
        cv2.imshow('src',src)
        cv2.waitKey(0)
        
        
# for cnt in contours: 
#     rect = cv2.boundingRect(cnt) 
#     x, y, w, h = rect 
#     if w*h > 40000: 
#      mask = np.zeros(src.shape, np.uint8) 
#      mask[y:y+h, x:x+w] = src[y:y+h, x:x+w] 
#      brightness = np.sum(mask) 
#      if brightness > max_brightness: 
#       brightest_rectangle = rect 
#       max_brightness = brightness 
#      cv2.imshow("mask", mask) 
#      cv2.waitKey(0) 

# x, y, w, h = brightest_rectangle 
# cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1) 
# cv2.imshow("canvas", canvas) 
# cv2.imwrite("result.jpg", canvas) 
# cv2.waitKey(0) 
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:08:16 2021

@author: Lenovo
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from video_process import *



video_path = r'C:/Users/Lenovo/Desktop/Workplace/round_new.avi'

video = Video(video_path)
print(video)

# video.capture.set(cv2.CAP_PROP_FRAME_COUNT,video.time2frame_idx(44))

# frame = video.get_frame_by_time(10)








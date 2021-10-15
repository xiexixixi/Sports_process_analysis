# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:36:20 2021

@author: Lenovo
"""

import os
import time
import cv2
import pickle

def show_time(func):
    def wrapped(*args):       
        start_time = time.time()
        func(*args)
        end_time = time.time()
        
        print("Costs %d seconds\n"%(end_time-start_time))
    return wrapped

class Video(object):
    
    def __init__(self,video_path = None):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.shape = (self.width, self.height)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frame = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration = int(self.total_frame//self.fps)
        
        
        sec = self.duration % 60
        minu = (self.duration//60)%60
        hr = self.duration // 3600
        self.duration_hms = (hr,minu,sec)
    
    def __len__(self):
        return int(self.total_frame)
    def __repr__(self):
        self.info = 'video path: {0}\nimage size: {1}\nfps:{2}\nframe num: {3}\
            \nduration(s): {4}s {5}\n'.format(self.video_path,\
                    (self.width,self.height),self.fps,self.total_frame,\
                        self.duration,self.duration_hms)
        return self.info

    
    def play(self,start_frame=0,end_frame=None):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        
        while(self.capture.isOpened()):
            ret, frame = self.capture.read()
            cv2.imshow('frame',frame)
            
            current_frame_pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)

            if (cv2.waitKey(1) & 0xFF == ord('q')) or \
                end_frame is not None and current_frame_pos>end_frame:
                break

                
        self.capture.release()
        cv2.destroyAllWindows()
    
    
    def _clip(self,save_path,start_frame=0,end_frame=None,\
              videoWriter=None,new_frame_size = None) -> bool:
        
        # 设置参数
        if end_frame is None or end_frame > self.total_frame:
            end_frame = self.total_frame
            
        assert start_frame < end_frame <= self.total_frame , "起止帧大小不对"
        if videoWriter is None:
            video_fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
            videoWriter = cv2.VideoWriter(save_path,video_fourcc,\
                                          self.fps,(self.width,self.height))
        
        # 提示视频时长
        print('start:%d\nend:%d\nduration:%ds\n'%(start_frame,end_frame,(end_frame-start_frame)/self.fps))
        
        
        # 开始写视频
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        while(self.capture.isOpened()):
            current_frame_pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)

            if current_frame_pos<end_frame:
                ret, frame = self.capture.read()
                
                # 改变frame大小
                if new_frame_size is not None:
                    frame = cv2.resize(frame, new_frame_size,interpolation=cv2.INTER_CUBIC)

                if ret:
                    videoWriter.write(frame)
                else:
                    print('视频解析错误')
                    return False
            else:
                return True
        return False
    
    @show_time
    def clip_by_time(self,save_path,start_time=0,duration=None,end_time=None,\
                     video_writer:cv2.VideoWriter=None,new_frame_size:tuple=None) -> bool:
        '''
        :param save_path: 保存视频路径
        :param start_time: 截取视频开始时间
        :param end_time: 截取视频结束时间
        :param video_writer: <cv2.VideoWriter> 
        :param new_frame_size: 新视频大小

        :return: <bool> 是否成功截取视频
        
        '''
        start_time = self.time2sec(start_time)
        
        if duration is not None:
            duration = self.time2sec(duration)
            end_time = start_time + duration
        else:
            end_time = self.time2sec(end_time)
            
        start_frame = self.time2frame_idx(start_time)
        end_frame = self.time2frame_idx(end_time)
        
        # print("expected time:%ds"%(0.324*duration + 2))
        return self._clip(save_path,start_frame,end_frame,video_writer,new_frame_size)
        
    def get_frame(self,frame_idx):
        if frame_idx > self.total_frame:
            print('frame_idx is greater than total frame')
            return None
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
        if self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                return frame
        return None
        
    
    def get_frame_by_time(self,time):
        
        return self.get_frame(self.time2frame_idx(time))
    
    @show_time
    def sample_class(self,root_path,time_seg:list,fps,class_name,down_sampling = 1):
        
        # 判断有无文件夹,没有则创建文件夹
        class_path = os.path.join(root_path,class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            print('Created dir:%s'%class_path)
        
        
        i = 0
        pic_num = len(os.listdir(class_path))
        for start_time,end_time in time_seg:
            start_frame = self.time2frame_idx(start_time)
            end_frame = self.time2frame_idx(end_time)

            for frame_idx in range(start_frame,end_frame,int(fps)):
                idx = i + pic_num
                frame_path = os.path.join(class_path,'%s_.%d.png'%(class_name,idx))
                cv2.imwrite(frame_path,self.get_frame(frame_idx)[::down_sampling,::down_sampling])
                i += 1
        else:
            print('Successfully write all %d images\n%d images in class \'%s\''%(i,idx+1,class_name))
            return True
        
        print('Not all expected pictures are written')
        return False
        
    @staticmethod
    def time2hms(time_by_sec):
        if type(time_by_sec) is tuple:
            return time_by_sec
        sec = time_by_sec % 60
        minu = (time_by_sec//60)%60
        hr = time_by_sec // 3600
        return hr,minu,sec
    
    @staticmethod
    def time2sec(hms):
        if type(hms) is not tuple:
            return hms
        return hms[0]*3600 + hms[1]*60 + hms[2]
    
    def frame_idx2time(self,frame_idx,natural = False):
        
        time = frame_idx/self.fps
        hr,minu,sec = self.time2hms(time)
        if natural:
            return int(hr),int(minu),int(sec)
        
        return hr,minu,sec

    def time2frame_idx(self,time:float or tuple):
        '''
        时间转换为帧，时间类型可以是float/int/tuple
        '''
        
        
        time = self.time2sec(time)
        frame_idx = int(time * self.fps)
        return frame_idx



    def show_video_info(self):
        print(self)
    



if __name__ == '__main__':

        
    video_path = r'D:/video/full/2019WTA迈阿密赛女单第2轮加西亚VS阿扎伦卡 英文录播.ts'
    video_path = r'D:/video/full/2018查尔斯顿赛第一轮布沙尔VS埃拉尼 回放.ts'

    
    # video_path = r'D:/video/clip.avi'  
    save_path = r'C:/Users/Lenovo/Desktop/Workplace/round_mpeg_shrink.avi'
    root_path = r'C:\Users\Lenovo\Desktop\Workplace\shit'
            
    start_time = (0,15,0)
    end_time = (0,15,5)
    
    video = Video(video_path)
    
    
    
    # 2018查尔斯顿赛第一轮布沙尔VS埃拉尼 回放.ts
    play_ground1 = [[(0,17,17),(0,17,19)],
                   [(0,18,20),(0,18,30)],
                   [(0,19,17),(0,19,28)],
                   [(0,23,36),(0,23,49)],
                   [(0,24,7),(0,24,32)],
                   [(0,40,32),(0,40,40)],
                   [(0,40,47),(0,40,51)],
                   [(0,41,14),(0,41,51)],
                   [(0,42,56),(0,43,14)],
                   [(0,55,10),(0,55,22)]]

    other1 = [[(0,2,43),(0,3,25)],
             [(0,3,37),(0,3,51)],
             [(0,4,3),(0,4,16)],
             [(0,46,15),(0,46,40)],
             [(0,53,8),(0,53,15)],
             [(0,53,54),(0,54,7)],
             [(0,54,47),(0,54,54)],
             [(1,14,53),(1,15,0)],
             [(1,41,36),(1,41,43)]]



    # 2019WTA阿卡普尔科站女单1_4决赛玛雅VS王雅繁 英文录播.ts
    play_ground2 = [[(0,11,20),(0,11,26)],
                    [(0,11,56),(0,12,6)],
                    [(0,12,28),(0,12,36)],
                    [(0,13,3),(0,13,17)],
                    [(0,13,42),(0,13,46)],
                    [(0,13,55),(0,14,12)],
                    [(0,14,42),(0,14,52)],
                    [(0,46,39),(0,46,44)],
                    [(0,47,15),(0,47,26)],
                    [(0,47,52),(0,47,59)],
                    [(0,48,51),(0,48,56)]]
    
    other2 = [[(0,0,0),(0,7,22)],
             [(0,11,33),(0,11,43)],
             [(0,36,00),(0,36,11)],
             [(0,37,37),(0,38,1)],
             [(0,44,6),(0,44,37)],
             [(0,52,45),(0,53,0)],
             [(1,1,27),(1,1,46)],
             [(1,6,14),(1,6,22)],
             [(1,15,30),(1,15,38)],
             [(1,20,19),(1,20,45)],
             [(1,21,10),(1,21,17)],
             [(1,51,19),(1,52,59)]]
   
    
    
    
    
    
    
    
    # 2019WTA迈阿密赛女单第2轮加西亚VS阿扎伦卡 英文录播.ts
    play_ground3 = [[(0,14,12),(0,14,18)],
                    [(0,14,40),(0,14,48)],
                    [(0,15,42),(0,15,46)],
                    [(0,16,1),(0,16,14)],
                    [(0,16,36),(0,16,47)],
                    [(1,29,44),(1,29,51)],
                    [(1,31,0),(1,31,16)]]
    
    other3 = [[(0,0,0),(0,7,43)],
              [(0,17,32),(0,19,17)],
              [(0,25,9),(0,25,17)],
              [(0,32,3),(0,32,24)],
              [(0,41,15),(0,41,30)],
              [(1,46,33),(1,48,5)]]

    

    # 2018查尔斯顿赛第一轮布沙尔VS埃拉尼 回放.ts
    # person = [[(0,7,22),(0,7,30)],
    #           [(0,17,1),(0,17,10)],
    #           [(0,18,46),(0,18,50)]]
    
    
    # other = [[(0,0,0),(0,0,23)],
    #           [(0,0,35),(0,1,25)],
    #           [(0,1,40),(0,1,50)],
    #           [(0,2,20),(0,2,37)],
    #           [(0,2,48),(0,2,51)],
    #           [(0,3,22),(0,3,25)],
    #          [(0,19,38),(0,19,50)]]
        
    
    # video.sample_class(root_path, back_ground, 25, 'back_ground',2)
    # video.sample_class(root_path, person, 25, 'person',2)
    # video.sample_class(root_path, other, 25, 'other',2)

    
    # play_ground=[]
    
    # play_ground.extend(back_ground1)
    # play_ground.extend(back_ground2)
    # play_ground.extend(back_ground3)
    
    dic1 = {}
    dic1['video_name'] = '2018查尔斯顿赛第一轮布沙尔VS埃拉尼 回放.ts'
    dic1['play_ground'] = play_ground1
    dic1['other'] = other1
    
    dic2 = {}
    dic2['video_name'] = '2019WTA阿卡普尔科站女单1_4决赛玛雅VS王雅繁 英文录播.ts'
    dic2['play_ground'] = play_ground2
    dic2['other'] = other2
    
    
    dic3 = {}
    dic3['video_name'] = '2019WTA迈阿密赛女单第2轮加西亚VS阿扎伦卡 英文录播.ts'
    dic3['play_ground'] = play_ground3
    dic3['other'] = other3 
    
    pickle.dump([dic1,dic2,dic3],open(r'./dics.pkl','wb'))
    

    # # video.play(start_frame,end_frame)
    
    # print(video)
    # new_frame_size = (video.width//2,video.height//2)
    # video_fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    # videoWriter = cv2.VideoWriter(save_path,video_fourcc,\
    #                               video.fps,new_frame_size)
        

    # # 剪切
    # video.clip_by_time(save_path,start_time,duration,video_writer=videoWriter,\
    #                     new_frame_size = new_frame_size)

    # video.clip_by_time(save_path,start_time,duration)

    # v1 =  Video(save_path)
    # v1.play()






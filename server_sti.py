# %%
from numba import jit
import pandas as pd
import numpy as np
import cv2
from numpy import mean
import function_server as func
import time
import matplotlib.pyplot as plt

''' socket '''
import socket
import time
import datetime


TCP_IP = "127.0.0.1"
TCP_PORT = 50112
BUFFER_SIZE = 512

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen() # it can be before loop

#%%
''' 影片的list '''
angle_li = [0]

''' dataframe '''
col = ['x_mm', "y_mm", "z_deg", "x_di", "y_di", "z_az", "r_x", "r_y", "r_z", "r_theta", "time"]
data = np.zeros([1,len(col)])

'''  '''
arpha = 0.01
square = 20
z_pixel = 100
send_server = 1


for i in range(len(angle_li)):
    print(i)

    '''new path'''
    video_name, this_video_theory_angle = func.select_video(5)
    video_name_s = video_name.strip('.avi')
    inputfilepath = 'C:/Users/ASUS/Desktop/生科/ming_algorithm/video/quick/' + video_name
    save_path = 'C:/Users/ASUS/Desktop/生科/ming_algorithm/8_camera_socket_arduino/result/'


    ''' get camera 葉 '''
    cap = cv2.VideoCapture(inputfilepath)
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    print(f"width: {WIDTH} ; height: {HEIGHT}")

    
    '''write video'''
    #writer_full = cv2.VideoWriter(save_path + video_name_s+'_video_all.avi',cv2.VideoWriter_fourcc(*'XVID'), FPS, (WIDTH*2, HEIGHT))
    #writer_crop = cv2.VideoWriter(save_path + video_name_s+'_video_crop.avi', cv2.VideoWriter_fourcc(*'XVID'), FPS, (square*3*5+60, square*3*2+30))
    
    ''' create dis 葉 '''
    dis = cv2.DISOpticalFlow_create(0)
    dis.setFinestScale(2)  # 2

    ''' prepare image '''
    ret, frame = cap.read()
    curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # bgr to grey
    prev = curr.copy()
    prev_crop = func.crop_frame(prev, WIDTH, HEIGHT)


    '''size of mag ang'''
    pre_obs = (0, 0, 0)
    crop_mag = np.zeros((5, 3*square, 3*square))
    crop_ang = np.zeros((5, 3*square, 3*square))

    
    ''' 等待server '''
    print('等待server')
    conn, addr = s.accept()
    print ('Connection address:', addr)
    time.sleep(5)

    '''迴圈 分析一個影片'''
    while(ret):

        ''' get image '''
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        ''' 全畫面分析開始 '''
        """
        frame_flow = dis.calc(prev, curr, None, )
        frame_mag, frame_ang = cv2.cartToPolar(frame_flow[..., 0], frame_flow[..., 1])  # ang 弧度

        # 顯示儲存影像
        #frame_full_combine = func.show_full_frame(frame, frame_ang, frame_mag, WIDTH, HEIGHT)
        #writer_full.write(frame_full_combine)
        """
        ''' 全畫面分析結束 '''
        
        

        time_start = time.clock()  # time

        ''' crop image 分析開始   '''
        curr_crop = func.crop_frame(curr, WIDTH, HEIGHT)
        for j in range(5):
            crop_flow = dis.calc(prev_crop[j], curr_crop[j], None, )
            crop_mag[j], crop_ang[j] = cv2.cartToPolar(crop_flow[..., 0], crop_flow[..., 1])  # ang 弧度

        # 顯示 儲存影像
        #frame_crop_combine = func.show_crop_frame(curr_crop, crop_ang, crop_mag)
        #writer_crop.write(frame_crop_combine )
        ''' crop image 分析結束   '''


        obs = func.get_rotation(crop_ang, crop_mag, WIDTH)
        new_obs = (obs[0]*(1-arpha) + pre_obs[0]*arpha, obs[1]*(1-arpha) + pre_obs[1]*arpha, obs[2]*(1-arpha) + pre_obs[2]*arpha)
        pre_obs = new_obs

        '''accumulate data'''
        z_az_deg = data[-1, 5] + new_obs[2]
        z_az_rad = np.deg2rad(z_az_deg)

        r = np.array(((np.cos(z_az_rad), -np.sin(z_az_rad)),
                      (np.sin(z_az_rad),  np.cos(z_az_rad))))
        
        v = np.array((new_obs[0], new_obs[1]))
        
        xy_di = r.dot(v)

        
        place = np.array([xy_di[0]+data[-1, 3], xy_di[1]+data[-1, 4]])
        ''' 判斷障礙物 '''
        """
        barrier_1 = (0, 10)
        barrier_1_d = np.sum((place-barrier_1)**2)**0.5
        if barrier_1_d < 5:
            vector = (place - barrier_1)/barrier_1_d*(5-barrier_1_d)
            place = place + vector
        """

        

        time_end = time.clock()  # time

        ''' 存新資料 '''
        #new_data = np.array([new_obs[0], new_obs[1], new_obs[2], xy_di[0]+data[-1, 3], xy_di[1]+data[-1, 4], z_az_deg, obs[3], obs[4], obs[5], obs[6], time_end - time_start]).reshape(1, 11)
        new_data = np.array([new_obs[0], new_obs[1], new_obs[2], place[0], place[1], z_az_deg, obs[3], obs[4], obs[5], obs[6], time_end - time_start]).reshape(1, 11)
        data = np.append(data, new_data, axis=0)

        ''' 備份和準備影像 '''
        prev = curr
        prev_crop = curr_crop
        ret, frame = cap.read()


        ''' socket send data '''
        if new_data[0,3]>0:
            s1 = str("1") + str(int(abs(new_data[0,3]*10))).rjust(5,'0')
        else:
            s1 = str("0") + str(int(abs(new_data[0,3]*10))).rjust(5,'0')

        if new_data[0,4]>0:
            s2 = str("1") + str(int(abs(new_data[0,4]*10))).rjust(5,'0')
        else:
            s2 = str("0") + str(int(abs(new_data[0,4]*10))).rjust(5,'0')
        
        if new_data[0,5]>0:
            s3 = str("1") + str(int(abs(new_data[0,5]*10))).rjust(5,'0')
        else:
            s3 = str("0") + str(int(abs(new_data[0,5]*10))).rjust(5,'0')


        signal_str = s1 + s2 + s3

        if send_server == 1 :
            s = time.clock()*1000
            #conn.send(str(datetime.datetime.now()).encode())
            conn.send(signal_str.encode())
            e = time.clock()*1000
            #print((e-s)*1000)
            print(f'send data :{signal_str}')
            # time.sleep(0.01)
        

        '''一個影片擬合結束'''


  
    # %%
    ''' dataframe '''
    data_df = pd.DataFrame(data, columns=col )
    data_df.to_csv("C:/Users/ASUS/Desktop/生科/ming_algorithm/6_video_socket/result/data.csv")
    print(data_df)

    


    '''釋放cap'''
    cap.release()
    #writer_full.release()
    #writer_crop.release()
    cv2.destroyAllWindows()


'''結束全部影片擬合'''
print('finish')


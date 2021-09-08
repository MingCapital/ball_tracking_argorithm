'''
https://github.com/basler/pypylon/blob/master/samples/opencv.py
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)
'''
# %%
# 相機
from pypylon import pylon
from pypylon import genicam

# 分析
import cv2
import numpy as np
import pandas as pd
import function_server as func
import time


# %%
# 設定參數  -----------------------------------------------
''' 相機 '''
WIDTH = 400     # 畫面寬度
HEIGHT = 400    # 畫面高度
GAIN = 1        # gain 越小越暗
OFFSETX = 192   # 畫面水平平移
OFFSETY = 91     # 畫面垂直平移

''' 影像 '''
arpha = 0.01    # 位移迭帶
SQUARE = 20     # DIS 大小

send_server = 1

save_path = 'C:/Users/ASUS/Desktop/生科/ming_algorithm/8_camera_socket_arduino/result/'
folder_name = "z_1800_2/" # 方向_轉速_次數


# 設定 dataframe -------------------------------------------
''' dataframe '''
col = ['x_mm', "y_mm", "z_deg", "x_di", "y_di", "z_az", "n_x", "n_y", "n_z", "n_theta", "delta_time", "time" ]
data = np.zeros([1,len(col)])

# opencv ---------------------------------------------------
''' writer 設定'''

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter( save_path + test_name + 'output.avi', fourcc, 20.0, (WIDTH, HEIGHT))
'''write video'''
FPS = 100
#writer_full = cv2.VideoWriter(save_path + folder_name + 'video_all.avi', cv2.VideoWriter_fourcc(*'XVID'), FPS, (WIDTH*2, HEIGHT))
#writer_crop = cv2.VideoWriter(save_path + folder_name + 'video_crop.avi', cv2.VideoWriter_fourcc(*'XVID'), FPS, (SQUARE*3*5+60, SQUARE*3*2+30))

''' DIS 設定 '''
dis = cv2.DISOpticalFlow_create(0)
dis.setFinestScale(2)  # DEFAULT =  2

# 相機 -----------------------------------------------------
''' 連接到第一個可用的相機 '''
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
'''轉換為opencv bgr格式'''
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

camera.Open()

''' 相機名稱 '''
print("Using device ", camera.GetDeviceInfo().GetModelName())
''' 演示一些功能訪問'''
print(f"相機名稱 : {camera.GetDeviceInfo().GetModelName()}")
print(f'camera.Width.GetInc()    = {camera.Width.GetInc()}')
print(f'camera.Width.GetValue()  = {camera.Width.GetValue()}')
print(f'camera.Height.GetInc()   = {camera.Height.GetInc()}') 
print(f'camera.Height.GetValue() = {camera.Height.GetValue()}')

print(f'camera.Gain.GetValue()   = {camera.Gain.GetValue()}')
print(f'camera.Gain.GetValue()   = {camera.Gain.GetValue()}')


''' 相機參數設定 '''
camera.Width.SetValue(WIDTH)
camera.Height.SetValue(HEIGHT)
camera.OffsetX.SetValue(OFFSETX)
camera.OffsetY.SetValue(OFFSETY)
camera.Gain.SetValue(GAIN)     

''' 要抓取的圖像數量'''
#countOfImagesToGrab = 5
#camera.StartGrabbingMax(countOfImagesToGrab) # 設定抓取數量
'''以最小延遲持續抓取（視頻）'''
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)



# 準備分析空間
''' 準備 crop 影像 '''
prepared_image = True
'''size of mag ang'''
pre_obs = (0, 0, 0)
crop_mag = np.zeros((5, 3*SQUARE, 3*SQUARE))
crop_ang = np.zeros((5, 3*SQUARE, 3*SQUARE))


while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded() and prepared_image==False :
        # 準備影像
        time_start = time.clock()  # time
        image = converter.Convert(grabResult)
        frame = image.GetArray()
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # bgr to grey

        ''' 全畫面分析開始 '''
        '''
        frame_flow = dis.calc(prev, curr, None, )
        frame_mag, frame_ang = cv2.cartToPolar(frame_flow[..., 0], frame_flow[..., 1])  # ang 弧度

        # 顯示儲存影像
        frame_full_combine = func.show_full_frame(frame, frame_ang, frame_mag, WIDTH, HEIGHT)
        writer_full.write(frame_full_combine)
        '''
        ''' 全畫面分析結束 '''

        

        ''' crop image 分析開始   '''
        time_end1 = time.clock()  # time
        curr_crop = func.crop_frame(curr, WIDTH, HEIGHT, SQUARE)
        for j in range(5):
            crop_flow = dis.calc(prev_crop[j], curr_crop[j], None, )
            crop_mag[j], crop_ang[j] = cv2.cartToPolar(crop_flow[..., 0], crop_flow[..., 1])  # ang 弧度
        # 顯示 儲存影像
        #frame_crop_combine = func.show_crop_frame(curr_crop, crop_ang, crop_mag)
        #writer_crop.write(frame_crop_combine )
        ''' crop image 分析結束   '''

        ''' 計算轉動 '''
        obs = func.get_rotation(crop_ang, crop_mag, WIDTH)
        new_obs = (obs[0]*(1-arpha) + pre_obs[0]*arpha, obs[1]*(1-arpha) + pre_obs[1]*arpha, obs[2]*(1-arpha) + pre_obs[2]*arpha)
        pre_obs = new_obs

        ''' 轉動積分 z->xy 開始 '''
        z_az_deg = data[-1, 5] + new_obs[2]   # z 積分
        z_az_rad = np.deg2rad(z_az_deg)
        r = np.array(((np.cos(z_az_rad), -np.sin(z_az_rad)),
                      (np.sin(z_az_rad),  np.cos(z_az_rad))))
        
        v = np.array((new_obs[0], new_obs[1]))
        
        xy_di = r.dot(v)

        place = np.array([xy_di[0]+data[-1, 3], xy_di[1]+data[-1, 4]]) # xy 積分
        ''' 轉動積分 z->xy 結束 '''

        time_end_all = time.clock()  # time
        #print(f'相機時間(ms) = {((time_end1 - time_start)*1000 ):5.2f},  光流時間(ms) = {((time_end_all - time_end1)*1000 ):5.2f}')

        ''' 存新資料 '''
        new_data = np.array([new_obs[0], new_obs[1], new_obs[2], place[0], place[1], z_az_deg, obs[3], obs[4], obs[5], obs[6], time_end_all - time_start, time_start_all - time_end_all ]).reshape(1, 12)
        data = np.append(data, new_data, axis=0)

        ''' 備份和準備影像 '''
        prev = curr
        prev_crop = curr_crop

        ''' 終止分析 '''
        if np.shape(data)[0]>2000:
            break
    else :
        # 準備影像
        print(f'準備分析影像')
        image = converter.Convert(grabResult)
        frame = image.GetArray()
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # bgr to grey

        cv2.imwrite(save_path + folder_name + 'check_image.jpg', curr)

        cv2.namedWindow('check_image', cv2.WINDOW_NORMAL)
        cv2.imshow('check_image', curr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f'影像確認完成')

        prev = curr.copy()
        prev_crop = func.crop_frame(prev, WIDTH, HEIGHT, SQUARE)
        prepared_image = False
        time_start_all = time.clock()  # time

    grabResult.Release()
    
# Releasing writer
#writer_full.release()
#writer_crop.release()

# Releasing the resource
camera.StopGrabbing()
cv2.destroyAllWindows()

print(f'關閉相機,結束分析')

data_df = pd.DataFrame(data, columns=col )
data_df.to_csv(save_path+ folder_name +"data.csv")

print(f'儲存資料')
# %%

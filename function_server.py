import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import jit
import cv2

''' 定義參數 '''
arpha = 0.01
square = 20
z_pixel = 100
middle_z_height = 320 # 中心的pixel高度 218

def select_video(video):
    if video == 1:
        return ('(540,0,0).avi',  [540/600 , 0, 0])

    elif video == 2:
        return ('(-180,0,0).avi', [-180/600, 0, 0])

    elif video == 3:
        return ('(0,540,0).avi',  [0, 540/600, 0])

    elif video == 4:
        return ('(0,-180,0).avi', [0, -180/600, 0])

    elif video == 5:
        return ('(0,0,540).avi',  [0, 0, 540/600])

    elif video == 6:
        return ('(0,0,-180).avi', [0, 0, -180/600])

    elif video == 7:
        return ('(5,0,-5)(360,0,0).avi', [0, 0, 0])





direction = ((0,0), (0,-z_pixel), (0,z_pixel), (-z_pixel,0), (z_pixel,0))
def crop_frame(image, WIDTH, HEIGHT, SQUARE):
    """
        z_p = z方向的pixel
    """
    global direction
    data_array = np.zeros((5, 3*SQUARE, 3*SQUARE), dtype='uint8')

    x0 = int(( WIDTH/2 - SQUARE*1.5))
    y0 = int((HEIGHT/2 - SQUARE*1.5))

    for i in range(5):
        data_array[i] = image[ y0+direction[i][1] : y0+direction[i][1]+SQUARE*3, x0+direction[i][0] : x0+direction[i][0]+SQUARE*3 ]
    return (data_array)


def plot_rectangle(frame, WIDTH, HEIGHT):
    """
    用在show_full_frame 裡面
    把要分析的部分框起來
    """
    global square, z_pixel, direction

    x0 = int(( WIDTH/2 - square*1.5))
    y0 = int((HEIGHT/2 - square*1.5))

    b = (255, 0, 0)  # blue
    g = (0, 255, 0)  # green
    r = (0, 0, 255)  # red
    w = (255, 255, 255)  # black

    for i in range(5):
        edge = (x0+direction[i][0], y0+direction[i][1]) 
        cv2.rectangle(frame, (edge[0], edge[1]),(edge[0]+3*square, edge[1]+3*square), w, 2)
        cv2.rectangle(frame, (edge[0]+square, edge[1]+square),(edge[0]+2*square, edge[1]+2*square), r, 2)

    return(frame)

def show_full_frame(frame, ang, mag, WIDTH, HEIGHT):
    """
    用ang mag 畫bgr
    畫框框，在把影像組合在一起
    再回傳
    """
    global square, z_pixel, direction

    hsv = np.zeros_like(np.zeros((HEIGHT, WIDTH, 3), dtype='uint8'))
    hsv[..., 1] = 255    # 藍綠紅
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    frame = plot_rectangle(frame, WIDTH, HEIGHT )
    bgr   = plot_rectangle(bgr  , WIDTH, HEIGHT )
    frame_combine = np.hstack((frame,  bgr))
    cv2.imshow('frame_combine', frame_combine)
    cv2.waitKey(1)

    return frame_combine

def get_bgr(ang, mag):
    """
    用在show_crop_bgr 裡面
    """
    global square

    frame = np.zeros((square*3, square*3, 3), dtype='uint8') # 高 寬
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255    # 藍綠紅
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def show_crop_frame(crop_frame, ang, mag):
    """
    顯示剪裁的影像
    """
    global square
    ''' 原始畫面 '''
    frame_v = np.zeros((3*square, 10, 3), dtype='uint8')  # b black
    frame_h = np.zeros((10, 3*square*5+60,  3), dtype='uint8')  # h horizontal

    ''' 原始畫面 '''
    frame_all = frame_v
    for i in range(5):
        color = cv2.cvtColor(crop_frame[i], cv2.COLOR_GRAY2BGR)
        frame_all = np.hstack((frame_all, color, frame_v))

    ''' hsv '''
    hsv_all = frame_v
    for i in range(5):
        hsv = get_bgr(ang[i], mag[i])
        hsv_all = np.hstack((hsv_all, hsv, frame_v))

    frame = np.vstack((frame_h, frame_all, frame_h, hsv_all, frame_h))
    cv2.imshow("frame ", frame)
    cv2.waitKey(1)
    return (frame)

def get_rotation(ang, mag, WIDTH):
    """
    用在crop_image 身上
    """
    global square ,z_pixel ,middle_z_height
    #f0 = 0.01297  # 4.8/370 # real
    f0 = 0.009375
    s = square 
    

    '''axis rotation'''
    z_co = (middle_z_height**2-z_pixel**2)**0.5
    v_0 = np.array([[0, 0, -middle_z_height],     #中     
                    [0, -z_pixel, -z_co], #上 
                    [0, z_pixel, -z_co],  #下
                    [-z_pixel, 0, -z_co], #左
                    [z_pixel, 0, -z_co]]) #右

    v_1 = np.zeros((5,3))


    ''' xy '''
    i = 0    # 中
    crop_ang = ang[i][s:s*2, s:s*2]
    crop_mag = mag[i][s:s*2, s:s*2]
    x_pixel = np.mean(np.sin(crop_ang) * crop_mag )
    y_pixel = np.mean(np.cos(crop_ang) * crop_mag )


    ''' z_middle '''
    z_new = -(middle_z_height**2 - x_pixel**2 - y_pixel**2)**0.5
    v_1[0] = np.array([ y_pixel, x_pixel, z_new ])
    
    
    ''' z_side '''
    z_cos_pixel = np.zeros(5) # 中上下左右
    z_sin_pixel = np.zeros(5) # 中上下左右

    for i in range(1,5): #上下左右
        crop_ang = ang[i][s:s*2, s:s*2]
        crop_mag = mag[i][s:s*2, s:s*2]
        z_cos_pixel[i] = np.mean(np.cos(crop_ang) * crop_mag)
        z_sin_pixel[i] = np.mean(np.sin(crop_ang) * crop_mag)
        z_new = -(middle_z_height**2 - (v_0[i][0]+z_cos_pixel[i])**2 - (v_0[i][1]+z_sin_pixel[i])**2)**0.5
        v_1[i] = np.array([ v_0[i][0]+z_cos_pixel[i], v_0[i][1]+z_sin_pixel[i], z_new ])

    z_ang = np.rad2deg(np.arcsin(((z_sin_pixel[4] - z_sin_pixel[3]) + (z_cos_pixel[1] - z_cos_pixel[2]))/4/z_pixel))

    '''算旋轉矩陣'''
    R = np.zeros((3,3))
    for i in range(3):
        y = v_1[:,i]
        x = v_0
        R[i] = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    
    M = R - R.T
    #print(new)
    a = M[2][1]
    b = M[0][2]
    c = M[1][0]
    d = (a**2 + b**2 + c**2)**0.5
    n = (a/d, b/d, c/d)
    #print('法向量: ', n )
    #print(np.sum(np.array(n**2) ))

    tr = np.trace(R)
    #print(tr)
    n_theta = np.rad2deg(np.arccos( (tr-1)*0.5 ))
    #print(beta, "   ", theta)
    #print(round(tr, 9), "   ", theta)

    
    return (x_pixel*f0, y_pixel*f0*(-1) ,z_ang, n[0], n[1], n[2], n_theta)
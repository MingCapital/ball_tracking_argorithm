import socket
import sys
import logging
import time
import cv2
import numpy as np


''' socket '''
HOST, PORT = "127.0.0.1", 50112
from time import sleep
data = " ".join(sys.argv[1:])

'''定義video 視覺刺激'''
fps = 60
video_name_led = 'client_led.mp4'
video_name_map = 'client_map.mp4'
save_path = 'C:/Users/ASUS/Desktop/生科/ming_algorithm/6_video_socket/result/'
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
led_writer = cv2.VideoWriter(save_path + video_name_led, fourcc, fps, (160, 32))
map_writer = cv2.VideoWriter(save_path + video_name_map, fourcc, fps, (1000, 1000))

'''arduino'''
import serial
COM_PORT = 'COM4'  # 請自行修改序列埠名稱
BAUD_RATES = 230400 #230400 # 500000 #2000000
ser = serial.Serial(COM_PORT, BAUD_RATES)
sleep(3)


def get_az(v1): #已經nor
    theta = np.rad2deg(np.arccos(v1[0]))
    if v1[1] >= 0 :
        az = theta
    else :
        az = 360 - theta
    return az

def cylinder_all_bd( dro, all_c ):
    """
    dro   果蠅的位置 
    all_c 圓柱 (目前是兩個)
    get 新的邊界
    再從新邊界算出新的刺激
    """
    all_bd = [0]* all_c.shape[0]           # 裝兩個圓柱的邊界

    # 算邊界
    for i, c1 in  enumerate ( all_c ):
        # cylinder initial condition 
        in_d = int( (c1[0]**2+c1[1]**2)**0.5 )
        in_h = c1[2]
        in_r = c1[3]

        # get delta_x, delta_y
        c1x, c1y = (c1[0]-dro[0], c1[1]-dro[1])
        d1 = (c1x**2 + c1y**2)**0.5     # get distazce
        v1 = (c1x/d1, c1y/d1)           # get unit vector
        a1 = get_az(v1)                 # get azimuth
        #print( f'a1={a1}' )
    
        # bar's new boundary
        m1 = int( (((a1- dro[2])/360)*160 + 1600)%160  ) #減掉果蠅的朝向 &nor
        h1 = int((in_d/d1) * in_h)        # new high
        r1 = int((in_d/d1) * in_r)        # new radius
        
        if h1>32:
            h1 = 32
        all_bd[i] = (m1+r1, m1-r1, h1)    # (左 右 高度)

    # 從邊界算刺激
    sti = np.zeros((32, 160, 3), dtype='uint8')
    for i in range(all_c.shape[0]):
        #(左邊界, 右邊界, 高度(已經判斷))
        height = all_bd[i][2]
        left   = 160 - all_bd[i][0]
        right  = 160 - all_bd[i][1]

        if left < 0 :
            sti[-height:, left: , 1] = 255
            sti[-height:, :right, 1] = 255

        elif right > 160 :
            sti[-height:, left:       , 1] = 255
            sti[-height:, :(right-160), 1] = 255

        else :
            sti[-height:, left:right, 1] = 255

    return(sti)


mask = np.array([[1],[2],[4],[8],[16],[32],[64],[128]])
def binary_2_bit(binary):
    bit = np.zeros((4, 160))
    for i in range(4):
        row = binary[i*8 : (i+1)*8,:]
        bit[i,:] = (row.T).dot(mask).T
    return(bit)

def different_bit(pre_bit, new_bit):
    judge = pre_bit != new_bit
    signal_str = ""
    for i in range(160):
        if (judge[:,i].any()):

            signal = (i//8, i%8+1, int(new_bit[0,i]), int(new_bit[1,i]), int(new_bit[2,i]), int(new_bit[3,i]))

            signal_str_1 = str(signal[0]).rjust(2,'0') + str(signal[1]) + str(signal[2]).rjust(3,'0') + str(signal[3]).rjust(3,'0') + str(signal[4]).rjust(3,'0') + str(signal[5]).rjust(3,'0')

            print(signal)
            signal_str += signal_str_1
            
            #sleep(0.00000000001)            # 暫停0.5秒，再執行底下接收回應訊息的迴圈
    signal_str += "\n"
    ser.write(signal_str.encode())  # 訊息必須是位元組類型
    print(signal_str)

'''定義圓柱 '''
in_x, in_y, in_h, in_r = (0, 80, 16, 8)  # 設定初始位置,高度,半徑 (mm, mm, num, num)
cylinder = np.array([[ in_x,  in_y, in_h, in_r ],
                     [ in_x, -in_y, in_h, in_r ]])

def initial_stimulus(bit):
    print('初始化是學刺激')
    signal_str = ""
    for i in range(160):
        signal = (i//8, i%8+1, int(bit[0,i]), int(bit[1,i]), int(bit[2,i]), int(bit[3,i]))

        signal_str_1 = str(signal[0]).rjust(2,'0') + str(signal[1]) + str(signal[2]).rjust(3,'0') + str(signal[3]).rjust(3,'0') + str(signal[4]).rjust(3,'0') + str(signal[5]).rjust(3,'0')

        print(signal)
        signal_str += signal_str_1
        
        #sleep(0.00000000001)            # 暫停0.5秒，再執行底下接收回應訊息的迴圈
    signal_str += "\n"
    ser.write(signal_str.encode())  # 訊息必須是位元組類型
    print(signal_str)

'''算虛擬的圓柱與圓心的距離'''
distance = (in_x**2 + in_y**2)**0.5

deg = in_r/160*360
tang = np.tan(np.deg2rad(deg))
sti_r = int(distance * tang)

middle = int(1000/2)

def dro_map(place):
    img = np.zeros((1000, 1000, 3), np.uint8)
    img.fill(100)
    global in_x, in_y, in_h, in_r, sti_r, middle

    cv2.circle(img,(middle-in_y, middle-in_x), sti_r, (0, 255, 0), -1)
    cv2.circle(img,(middle+in_y, middle-in_x), sti_r, (0, 255, 0), -1)


    '''果蠅位置'''
    dro_x = int(place[0])
    dro_y = int(place[1])

    rad = np.deg2rad(place[2])

    cv2.line(img, (middle-dro_y, middle-dro_x), (int(middle-dro_y-np.sin(rad)*10),int( middle-dro_x-np.cos(rad)*10)), (0, 0, 255), 2)
    cv2.circle(img,(middle-dro_y, middle-dro_x), 3, (0, 0, 0), -1)
    
    return img


'''視覺刺激'''
sti = np.zeros((32, 160, 3), dtype='uint8')

place = (0,0,0)
pre_sti = cylinder_all_bd(place, cylinder)
pre_sti_g = pre_sti[:,:,1]/255
pre_bit = binary_2_bit(pre_sti_g)
led_writer.write(pre_sti)
initial_stimulus(pre_bit)


map = dro_map(place)
map_writer.write(map)




''' main function '''
for x in range(100):
    # Create a socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        
        # Connect to server and send data
        sock.connect_ex((HOST, PORT))
        # receive data periodically - it should run in thread
        while True:
            received = str(sock.recv(1800), "utf-8")
            print(f'get data:{received[-18:]}')

            sign = (int(received[-18]), int(received[-12]), int(received[-6])  )
            sign_list= [1, 1, 1]
            for i, element in enumerate(sign) :
                if element==0 :
                    sign_list[i] = -1

            place = (float(received[-17:-12])/10* sign_list[0], float(received[-11:-6])/10* sign_list[1], float(received[-5:])/10* sign_list[2])
            #print(f'get data:{place}')


            ''' 新視覺刺激 '''
            new_sti = cylinder_all_bd(place, cylinder)
            led_writer.write(new_sti)


            '''換bit 傳訊號到 arduino'''
            new_sti_g = new_sti[:,:,1]/255
            new_bit = binary_2_bit(new_sti_g)

            different_bit(pre_bit, new_bit)
            


            ''' 存影像 '''
            pre_sti = new_sti
            pre_sti_g = new_sti_g
            pre_bit = new_bit


            ''' map '''
            map = dro_map(place)
            map_writer.write(map)




led_writer.release()
map_writer.release()
cv2.destroyAllWindows()
print('finish')

sock.close()
logging.info("Socket closed")
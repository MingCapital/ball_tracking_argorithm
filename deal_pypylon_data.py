# %%
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np



def plot_one_video_result(data_df, the_time = 0, folder_name = '', path=''):
        """
        this_video_theory_angle    理論轉動角度
        data_df                   計算角度
        """
        x_axis = np.linspace(1, len(data_df["x_mm"]), len(data_df["x_mm"]))
        x_axis_1 = np.ones(len(data_df["x_mm"]))

        tsize = 20
        lsize = 15
        ssize = 15
        alpha_1 = 0.15
        alpha_2 = 0.7

        fig, axes = plt.subplots(3,3, figsize=(18,20))
        fig.suptitle(folder_name, fontsize=30 )
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        '''x displacement'''
        plt.subplot(map_v, map_h, 1)
        plt.plot(x_axis, data_df["x_mm"], c='b', label="x_result", alpha=alpha_2)
        plt.plot(x_axis, data_df["y_mm"], c='m', label="y_result", alpha=alpha_1)
        plt.title('displacement_x', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("displacement (mm)", fontsize=lsize)
        plt.legend()

        '''y displacement'''
        plt.subplot(map_v, map_h, 2)
        plt.plot(x_axis, data_df["x_mm"], c='b', label="x_result", alpha=alpha_1)
        plt.plot(x_axis, data_df["y_mm"], c='m', label="y_result", alpha=alpha_2)
        plt.title('displacement_y', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("displacement (mm)", fontsize=lsize)
        plt.legend()

        '''z displacement'''
        plt.subplot(map_v, map_h, 3)
        plt.plot(x_axis, data_df["z_deg"], c='g', label="z_result", alpha=alpha_2)
        plt.title('angle_z', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("angle (d)", fontsize=lsize)

        '''x location'''
        plt.subplot(map_v, map_h, 4)
        plt.plot(x_axis, data_df["x_di"], c='b', label="x_result", alpha=alpha_2)
        plt.plot(x_axis, data_df["y_di"], c='m', label="y_result", alpha=alpha_1)
        plt.title('location_x', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("location (mm)", fontsize=lsize)
        plt.legend()

        '''y location'''
        plt.subplot(map_v, map_h, 5)
        plt.plot(x_axis, data_df["x_di"], c='b', label="x_result", alpha=alpha_1)
        plt.plot(x_axis, data_df["y_di"], c='m', label="y_result", alpha=alpha_2)
        plt.title('location_y', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("location (mm)", fontsize=lsize)
        plt.legend()


        '''z azimuth'''
        plt.subplot(map_v, map_h, 6)
        plt.plot(x_axis, data_df["z_az"], c='g', label="z_result", alpha=alpha_2)
        plt.title('azimuth', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("angle (d)", fontsize=lsize)
    
        ''' map '''
        plt.subplot(map_v, map_h, 7)
        plt.plot(data_df["x_di"], data_df["y_di"], c='k', label="x_result", alpha=alpha_2)

        plt.title('map', fontsize=tsize)
        plt.xlabel("x_location (mm)", fontsize=lsize)
        plt.ylabel("y_location (mm)", fontsize=lsize)

        ''' n_theta '''
        plt.subplot(map_v, map_h, 8)
        plt.plot(x_axis, data_df["n_theta"], c='k', label="z_result", alpha=alpha_2)
        plt.title('n_theta', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("angle (d)", fontsize=lsize)
        

        '''z anngular velocity'''
        plt.subplot(map_v, map_h, 9)
        plt.plot(x_axis, data_df["z_deg"]/data_df["delta_time"], c='g', label="z_result", alpha=alpha_2)
        plt.plot(x_axis, x_axis_1*360/the_time, c='k', label="z_theory", alpha=alpha_2, lw=3)
        plt.title('anngular velocity', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("Angular velocity (d/s)", fontsize=lsize)

        ''' n_x '''
        plt.subplot(map_v, map_h, 10)
        plt.plot(x_axis, data_df["n_x"], c='b', label="z_result", alpha=alpha_2)
        plt.title('n_x', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("n_x", fontsize=lsize)

        ''' n_y '''
        plt.subplot(map_v, map_h, 11)
        plt.plot(x_axis, data_df["n_y"], c='m', label="z_result", alpha=alpha_2)
        plt.title('n_y', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("n_y", fontsize=lsize)

        ''' n_z '''
        plt.subplot(map_v, map_h, 12)
        plt.plot(x_axis, data_df["n_z"], c='g', label="z_result", alpha=alpha_2)
        plt.title('n_z', fontsize=tsize)
        plt.xlabel("frame", fontsize=lsize)
        plt.ylabel("n_z", fontsize=lsize)
    



        plt.savefig(path + '位移.png', bbox_inches='tight',transparent=True, edgecolor='b')
        plt.show()
        plt.close()

#file_list = ['z_900_2/','z_1200_2/','z_1500_2/','z_1800_2/']

file_list = ['z_900_1/','z_1200_1/','z_1500_1/','z_1800_1/']
time_list = np.array([3.701, 4.923, 6.148, 7.384])




# %%
save_path = 'C:/Users/ASUS/Desktop/生科/ming_algorithm/8_camera_socket_arduino/result/'

for i in range(len(file_list)):
    map_v = 4
    map_h = 3
    folder_name = file_list[i]
    the_time = time_list[i]
    df = pd.read_csv(save_path + folder_name + 'data.csv', engine='python') 
    df.fillna(0) # nan20
    plot_one_video_result(df, the_time, folder_name, save_path + folder_name)


# %%
''' z 旋轉比較 '''
''' 算角速度 '''
time_list = np.array([3.701, 4.923, 6.148, 7.384])
the_angular_velocity = 360/time_list
exp_angular_velocity = np.zeros(4)
std_angular_velocity = np.zeros(4)

for i in range(len(file_list)):
    folder_name = file_list[i]
    the_time = time_list[i]
    df = pd.read_csv(save_path + folder_name + 'data.csv', engine='python')
    df.fillna(0) # nan20


    exp_1 = np.mean(df["z_deg"]/df["delta_time"])
    std_1 = np.std( df["z_deg"]/df["delta_time"])

    exp_angular_velocity[i] = exp_1
    std_angular_velocity[i] = std_1

# %% 
x = [1, 2, 3, 4]
plt.plot(x, the_angular_velocity, label = 'theory' )
plt.errorbar(x, exp_angular_velocity, std_angular_velocity, linestyle='None', marker='^', label = 'test')
plt.title("z rotation", size = 25)
plt.xlabel("number",size = 20)
plt.ylabel("angular velocity (d/s)",size = 20)
plt.legend()
plt.show()
    


# %%

import math
import numpy as np
import time
from pure_pursuit.utils import car_controllor, VehicleState, pure_pursuit_control

'''
主程序
'''
if __name__ == '__main__':
    
    '''
    路线跟踪常量
    '''
    Lf = 200.0  # look-ahead distance
    dt = 1.0  # [s]
    d = 144.0  # [mm] wheel base of vehicle
    delta_candicate = np.array([-28, -16.2, 0.0, 16.2, 28])/180*np.pi
    delta_cmd = np.array([90, 60, 0, -60, -90], np.int16) + 90

    path_y = list(range(0, 2100, 300))
    path_x = [math.sin(ix/300) * 400.0 for ix in path_y]
    path = np.array([path_x, path_y]).T

    # path_x = [0] * 10
    # path_y = list(range(0, 3000, 300))
    # path = np.array([path_x, path_y]).T

    # path_x = [0, 0, 0, 0, 0, -300, -600, -900, -1200]
    # path_y = [0, 300, 600, 900, 1200, 1200, 1200, 1200, 1200]
    # path = np.array([path_x, path_y]).T / 3*2

    print(path)

    ctrller = car_controllor()
    ctrller.servo_control(90)
    ctrller.motor_control(50)
    time.sleep(2)
    state = VehicleState(x=0, y=0, delta=0.0, yaw=0, v=0)

    pure_pursuit_control(state, path, ctrller)
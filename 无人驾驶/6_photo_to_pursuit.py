import torch
from PIL import Image

from seg import unet_predict
from seg.unet import UNet

from path_plan.utils import find_path, printpaths, smooth

import numpy as np
import matplotlib.pyplot as plt
from pure_pursuit.utils import car_controllor, VehicleState, pure_pursuit_control

'''
主程序
'''
if __name__ == '__main__':

    SHOW_IMAGE = True
    DIM = (800, 600)

    net = UNet(n_channels=3, n_classes=3, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load('./seg/checkpoints/0711/checkpoint_epoch2.pth', map_location=device))

    img = Image.open('./test_image/train_43.jpg')
    mask = unet_predict.predict_img(net=net,
                        full_img=img,
                        scale_factor=0.5,
                        out_threshold=0.5,
                        device=device)
    mask = unet_predict.mask_to_image(mask)

    plt.imshow(mask)
    plt.show()

    path = find_path(mask, time_limit=5, SHOW_IMAGE=True)
    smooth_path = smooth(path, weight_data=0.5, weight_smooth=0.3)

    path_array = np.array(path)
    smooth_path_array = np.array(smooth_path, np.int16)

    path_array = np.vstack((path_array[:, 1], path_array[:, 0])).T
    smooth_path_array = np.vstack(\
        (smooth_path_array[:, 1], smooth_path_array[:, 0])
    ).T

    path_array[:, 1] = 29 - path_array[:, 1]
    smooth_path_array[:, 1] = 29 - smooth_path_array[:, 1]

    path_array = path_array * 40
    smooth_path_array = smooth_path_array * 40

    print(path_array)


    plt.figure()
    plt.scatter(path_array[:, 0], path_array[:, 1], s=20, c='red', alpha=0.6)
    plt.scatter(smooth_path_array[:, 0], smooth_path_array[:, 1], s=20, c='green', alpha=0.6)
    plt.show()


    '''
    路线跟踪常量
    '''
    Lf = 100.0  # look-ahead distance
    dt = 1.0  # [s]
    d = 144.0  # [mm] wheel base of vehicle
    delta_candicate = np.array([-28, -16.2, 0.0, 16.2, 28]) / 180 * np.pi
    delta_cmd = np.array([90, 60, 0, -60, -90], np.int16) + 90

    ctrller = car_controllor()
    ctrller.servo_control(90)
    ctrller.motor_control(50)
    state = VehicleState(x=800, y=0, delta=0.0, yaw=0, v=0)

    # 1 px = 40 mm
    pure_pursuit_control(state, path_array, ctrller)
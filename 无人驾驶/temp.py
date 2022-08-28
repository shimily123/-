import requests
import time
import numpy as np

# d = 144.0  # [mm] wheel base of vehicle
# delta_candicate = np.array([-28, -16.2, 0.0, 16.2, 28])/180*np.pi
# delta_cmd = np.array([90, 60, 0, -60, -90], np.int16) + 90

# delta = delta_candicate[0]
# R = d / np.tan(delta)
# print(R)

# r = requests.post('http://192.168.4.1/motor_control?speed=50')
# r = requests.post('http://192.168.4.1/servo_control?angle=180')

r = requests.post('http://192.168.4.1/motor_control?speed=50')
time.sleep(5)
r = requests.get('http://192.168.4.1/read_speed')
print(r.text)
time.sleep(5)
r = requests.post('http://192.168.4.1/motor_control?speed=0')
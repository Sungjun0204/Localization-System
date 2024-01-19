#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String
from scipy.optimize import least_squares
import numpy as np
import time
import matplotlib.pyplot as plt
import levenberg_marquardt as LM
import subprocess
import pprint


# 전역변수 선언
zero_setting_flag = 0           # 4~5번 뒤에 offset을 작동시키기 위한 flag변수 선언
first_value = np.array([0.118, 0.118, 0.65, 0.877, 0.25, 0.671]) # 0~2번째는 위치 초기 값, 3~5번째는 자계강도(H) 초기 값
full_packet = ""                # 패킷 값을 저장하는 리스트 변수
sensor_data = []                # 해체작업을 진행할 패킷 값을 저장하는 리스트 변수
packet_count = 0                       # 분할되어 들어오는 패킷을 총 10번만 받게 하기 위해 카운트를 세는 변수
is_collecting = False

array_Val = np.array([  [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0] ])     # sensor 값들의 numpy조작을 간편하게 하기 위해
                                         # 옮겨 저장할 배열 변수 선언
zero_Val = np.array([   [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0] ])     # offset을 위해 배경 노이즈값을 저장할 배열 변수 선언

P = np.array([                           # hall sensor의 각 위치좌표 값 초기화(좌표 단위는 mm)
                [-118,  118,   0],
                [   0,  118,   0],
                [ 118,  118,   0],
                [-118,    0,   0],
                [   0,    0,   0],      ## 5번 센서로, 중앙이 되는 센서다
                [ 118,    0,   0],
                [-118, -118,   0],
                [   0, -118,   0],
                [ 118, -118,   0] ]) #* (10**-3) # m 단위로 맞추기 위한 환산

# 상수 선언
MU0 = 4*(np.pi)*(10**-7)    # 진공투자율[uH/A]
MU = 1.056                  # 사용하는 자석의 매질 투자율[-]
M0 = 1320                   # 사용하는 자석의 등급에 따른 값[mT]: 실험에서 사용하는 자석 등급은 N42
M_T = (np.pi)*(0.0047625**2)*(0.0127)*M0    # 자석의 자화벡터 값 = pi*(반지름^2)*(높이)*(자석 등급)









# 함수 선언

# Serial_example_node.cpp를 통해 받은 패킷을 처리하는 함수
def seperating_Packet(data):
    global full_packet, packet_count, is_collecting, sensor_data, array_Val

    # 'Z'로 시작하는 패킷을 감지하거나 이미 패킷 수집 중인 경우
    if data.data.startswith('Z') or is_collecting:
        full_packet += data.data
        packet_count += 1

        # 패킷 수집 시작
        if data.data.startswith('Z'):
            is_collecting = True

        # 10개의 패킷을 수집한 경우
        if packet_count == 10:
            array_Val = parse_packet(full_packet)   # 배열 변수 array_Val에 센서 값 저장
            array_Val = np.array(array_Val)         # 리스트 변수를 다시 배열 변수로 변환
            # pretty_print(array_Val)
            if (zero_setting_flag == 0):  # flag변수가 0이면 offset 함수 호출
                zero_setting()

            offset_Setting()              # offset 설정하는 함수 호출

            full_packet = ""        # 패킷 초기화
            packet_count = 0        # 분할 패킷 카운트 초기화
            is_collecting = False   # 분할 패킷 저장 flag 초기화
            
    


#패킷 분해 함수에서 부호를 결정해주는 함수 
def parse_value(value_str):
    sign = -1 if value_str[0] == '1' else 1
    return sign * int(value_str[1:])

# 2차원으로 저장되어 있는 리스트 변수 깔끔하게 출력해주는 함수
def pretty_print(data):
    for row in data:
        print(" ".join("{:>10.2f}".format(x) for x in row))
    print('-----------------------------------------')

# 본격적인 센서 패킷 값 분해 코드
def parse_packet(packet):
    if not packet.startswith("ZZ"):
        return 0

    raw_sum = 0  # 원본 센서 값들의 합
    sensors_data = []
    for char in range(ord('A'), ord('I') + 1):
        start_char = chr(char)
        end_char = chr(char + 32)
        start_idx = packet.find(start_char) + 1
        end_idx = packet.find(end_char)
        sensor_str = packet[start_idx:end_idx]

        sensor_values = [parse_value(value) for value in sensor_str.split(',')]
        raw_sum += sum(sensor_values)  # 가공 전 원본 데이터 합산
        sensor_values[0] *= -1
        sensor_values = [v / 1000000.0 for v in sensor_values]  # UART통신을 위해 없앴던 소수점 부활 (x100)
                                                                # hall seneor는 단위가 uT이므로, mT로 단위 통일 (x100)
                                                                # 따라서 1,000,000을 나눠준다
        sensors_data.append(sensor_values)

    checksum_str = packet[packet.find('i') + 1:packet.find('Y')]
    checksum = parse_value(checksum_str)

    if raw_sum != checksum:
        return 0

    #pretty_print(sensors_data)
    return sensors_data





# 단순히 offset 설정을 위한 flag 변수 값을 조정하는 함수
def callback_offset(data):
    global zero_setting_flag
    zero_setting_flag = 0       # 다시 0으로 설정해서 offset 설정하는 함수 호출되게 한다.


# offset을 위한 배경 노이즈 값을 리스트에 저장하는 함수
def zero_setting():
    global array_Val
    global zero_Val
    global zero_setting_flag

    zero_Val = array_Val               # 센서에서 측정하는 배경 노이즈 값을 저장
    zero_Val = np.array(zero_Val)      # 리스트에서 배열 변수로 변환

    zero_setting_flag = 1                   # 이후 1로 설정하여 함수가 호출되지 않도록 초기화
                                            # 이렇게 안 하면 센서 측정 값이 계속 0으로만 출력됨


# offset 적용하는 함수
def offset_Setting():
    global array_Val, zero_Val, first_value
    result = []

    #command = "clear"
    #subprocess.call(command, shell=True)

    array_Val = np.array(array_Val) - np.array(zero_Val)    # offset 적용
    
    ### 본격적인 위치추정 코드 ###
    initial_guess = first_value    # 초기 자석의 위치좌표 및 자계강도 값

    result_pos = least_squares(residuals, initial_guess)
    #print(result_pos[0][0], result_pos[0][1], result_pos[0][2])    # leastsq 함수일 때의 출력
    
    #print(result_pos.x[0], result_pos.x[1], result_pos.x[2])        # least_squares 함수일 때의 위치값 출력
    #print(result_pos.x[3], result_pos.x[4], result_pos.x[5])       # least_squares 함수일 때의 H값 출력
    result = [result_pos.x[0], result_pos.x[1], result_pos.x[2]]
    pprint.pprint(result)
    
    # 위치추정을 위한 초기 위치값을 이전에 구한 위치로 초기화
    for i in range(3):
        first_value[i] = result_pos.x[i]


    # pretty_print(array_Val)   # 센서링된 값 출력 





# 자석의 자기밀도를 계산하는 함수 
# A: 자석의 현재 위치좌표, P: 센서의 위치좌표, H: 자석의 자계강도
def cal_B(A, P):
    global MU, M_T
    N_t = (MU * M_T) / (4*(np.pi))              # 상수항 계산 
    b1 = (3 * np.dot([A[3], A[4], A[5]], P) * P) / (distance_3d(P,[A[0], A[1], A[2]]) ** 5)    # 첫째 항 계산
    b2 = [A[3], A[4], A[5]] / (distance_3d(P,[A[0], A[1], A[2]]) ** 3)                         # 둘째 항 계산
    
    final_B = N_t * (b1 - b2)                      # 최종 자기밀도 값 계산
    return final_B                                 # 최종 자기밀도 값 반환

# 두 점 사이의 거리를 구하는 함수
def distance_3d(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)**0.5



# 측정한 자기장 값과 계산한 자기장 값 사이의 차이를 계산하는 함수
def residuals(init_pos):
    global array_Val, P, first_value
    differences = [0,0,0] # 센서 값과 계산 값 사이의 잔차 값을 저장하는 배열변수 선언

    # 위치에 대한 잔차 값의 총합 저장
    for i in range(9):
        buffer_residual = (array_Val[i] - (cal_B(init_pos, P[i])))
        #print(buffer_residual)
        #print("----") 
        differences[0] += buffer_residual[0]    # 각 센서들의 잔차를 각 축 성분끼리 더한다
        differences[1] += buffer_residual[1]
        differences[2] += buffer_residual[2]


    #pprint.pprint(differences)
    return differences






# 메인 함수
def main():

    global array_Val


    rospy.init_node('algorithm_pkg_node', anonymous=True)   # 해당 노드의 기본 설정
    rospy.Subscriber('read', String, seperating_Packet)   # /read를 구독하고 seperating_Packet 함수 호출: 패킷 처리 함수
    rospy.Subscriber('Is_offset', String, callback_offset)  # /Is_offset을 구독하고 callback_offset 함수 호출


    rospy.spin()    # node 무한 반복






if __name__ == '__main__':
    main()                  # main문 무한 호출
    

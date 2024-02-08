#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String
from scipy.optimize import least_squares
import numpy as np
import pprint


# 전역변수 선언
zero_setting_flag = 0           # 4~5번 뒤에 offset을 작동시키기 위한 flag변수 선언
first_value = np.array([0.118, -0.118, 0.065, 0.30, 0.30]) # 0~2번째는 위치 초기 값, 3~5번째는 자계강도(H) 초기 값
#first_value = np.array([118, -118, 65, 0.33, 0.33])
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
                [ 118, -118,   0] ]) * (1e-3) # m 단위로 맞추기 위한 환산

# 상수 선언
MU0 = 4*(np.pi)*(1e-7)    # 진공투자율[H/m]
MU = 1.0                    # 사용하는 자석의 매질 투자율[-]
M0 = 1.320 / MU0             # 사용하는 자석의 등급에 따른 값[A/m]: 실험에서 사용하는 자석 등급은 N42
                             # 1.3[T]에서 [A/m]로 환산하기 위해 MU0 값을 나눔
#M_T = (np.pi)*(4.7625**2)*(12.7)*M0    # 자석의 자화벡터 값 = pi*(반지름^2)*(높이)*(자석 등급)
M_T = (np.pi)*(0.0047625**2)*(0.0127)*M0

np.set_printoptions(precision=5, suppress=True)    # 배열 변수 출력 시 소수점 아래 5자리까지만 출력되도록 설정






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
        print(" ".join("{:>10.6f}".format(x) for x in row))
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
        sensor_values[0] *= -1         # 모든 X 좌표에 -1을 곱하여 좌표계를 오른손 기준으로 변경
        sensor_values = [v / 10 for v in sensor_values]  # UART통신을 위해 없앴던 소수점 부활 (/100)
                                                                # hall seneor는 단위가 uT이므로, mT로 단위 통일 (/1000)
                                                                # 만약 측정 단위가 nT라면 (/1,000,000) 연산을 해 줘야 함
                                                                # 따라서 100,000을 나눠준다
        sensors_data.append(sensor_values)

    checksum_str = packet[packet.find('i') + 1:packet.find('Y')]
    checksum = parse_value(checksum_str)

    if raw_sum != checksum:
        return 0

    # pretty_print(sensors_data)  # 패킷에서 분리한 raw data 값 확인
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
    result = [0, 0, 0]

    ### 출력 log 없이 한 줄로만 출력하고 싶을 때 사용 ###
    ## command = "clear"
    ## subprocess.call(command, shell=True)

    array_Val = np.array(array_Val) - np.array(zero_Val)    # offset 적용
    
    ### 본격적인 위치추정 코드 ###
    initial_guess = first_value    # 초기 자석의 위치좌표 및 자계강도 값
    bounds = ([-0.120, -0.120, 0, -10, -10, -10],[0.120, 0.120, 0.1, 10, 10, 10])   # initial_guess의 제약 설정

    result_pos = least_squares(residuals, initial_guess, method='lm')    # Levenberg-Marquardt Algorithm 계산
    
    result = np.array([result_pos.x[0], result_pos.x[1], result_pos.x[2]])  # 위치 근사값만 따로 저장
    pprint.pprint(result * 1000)                          # 위치 근사값 출력
    
    # 위치추정을 위한 초기값을 이전에 구한 추정값으로 초기화
    for i in range(5):
        first_value[i] = result_pos.x[i]


    #pretty_print(array_Val)   # 가공된 센서링된 값 출력 



# 측정한 자기장 값과 계산한 자기장 값 사이의 차이를 계산하는 함수
# (여기서 오차 제곱까지 해 줄 필요는 없음. least_squares에서 알아서 계산해 줌)
def residuals(init_pos):
    global array_Val, P, first_value
    differences = [0,0,0] # 센서 값과 계산 값 사이의 잔차 값을 저장하는 배열변수 초기화
    val =  [[array_Val[0],array_Val[1],array_Val[2]],
            [array_Val[3],array_Val[4],array_Val[5]],
            [array_Val[6],array_Val[7],array_Val[8]]]        # 센서 값을 3x3 형태로 다시 저장(for 계산 용이)
    k_ij = []     # K(i,j) 값을 저장할 리스트 변수 초기화
    hh = 0.118       # 센서들 사이 떨어져있는 거리

    # K_ij 값 계산
    for i in range(3):
        for j in range(3):
            param = [0,0,0,0,0]
            param[0]=0 if j-1 < 0 else val[i][j-1][2]
            param[1]=0 if j+1 > 2 else val[i][j+1][2]
            param[2]=0 if i-1 < 0 else val[i-1][j][2]
            param[3]=0 if i+1 > 2 else val[i+1][j][2]
            param[4]=(-4)*(val[i][j][2])

            k_ij.append( (-1)*(sum(param) / (hh**2)) )
    
    # 위치에 대한 잔차 값의 총합 저장
    for i in range(9):
        buffer_residual = k_ij[i] - cal_BB(init_pos, P[i])  # 실제값과 이론값 사이의 잔차 계산
        differences += buffer_residual    # 각 센서들의 잔차를 각 축 성분끼리 더한다

    #pprint.pprint(differences) # 계산한 잔차 값의 총합 출력
    return differences


# 자석의 자기밀도를 계산하는 함수 
# A: 자석의 현재 위치좌표, P: 센서의 위치좌표, H: 자석의 자계강도
def cal_BB(A_and_H, P):
    global MU0
    A = [A_and_H[0], A_and_H[1], A_and_H[2]]    # 위치 값 따로 A 리스트에 저장
    M = [A_and_H[3], A_and_H[4]]; M.insert(0, 1-(M[0]**2)-(M[1]**2))    # 자계강도 값 따로 H 리스트에 저장
    R = np.array(A-P)

    const = MU0 / (4*np.pi)     # 상수항 계산
    b1 = (9*M[2]) / (np.linalg.norm(R) ** 5)
    b2 = (45*(R[2])*(np.dot(M,R)+(M[2]*R[2]))) / (np.linalg.norm(R) ** 7)
    b3 = (105*(R[2]**3)*(np.dot(M,R))) / (np.linalg.norm(R) ** 9)

    return const*(b1-b2+b3)


# 두 점 사이의 거리를 구하는 함수
def distance_3d(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)**0.5






# 메인 함수
def main():

    global array_Val


    rospy.init_node('algorithm_pkg_node', anonymous=True)   # 해당 노드의 기본 설정
    rospy.Subscriber('read', String, seperating_Packet)   # /read를 구독하고 seperating_Packet 함수 호출: 패킷 처리 함수
    rospy.Subscriber('Is_offset', String, callback_offset)  # /Is_offset을 구독하고 callback_offset 함수 호출


    rospy.spin()    # node 무한 반복


if __name__ == '__main__':
    main()                  # main문 무한 호출
    

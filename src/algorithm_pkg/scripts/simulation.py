#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from scipy.optimize import least_squares
import numpy as np
import time
import subprocess
import pprint


# 전역변수 선언
zero_setting_flag = 0           # 4~5번 뒤에 offset을 작동시키기 위한 flag변수 선언
#first_value = np.array([0.118, -0.118, 0.065]) # 위치 초기 값
first_value = [0.118, -0.118, 0.065, 800, 800, 0]
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
                        [0, 0, 0] ])     # sensor 값들의 numpy조작을 간편하게 하기 위해 옮겨 저장할 배열 변수 선언


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
MU = 1.0                  # 사용하는 자석의 매질 투자율[-]
M0 = 1.320 / MU0          # 사용하는 자석의 등급에 따른 값[A/m]: 실험에서 사용하는 자석 등급은 N42 / 1.3[T]에서 [A/m]로 환산하기 위해 MU0 값을 나눔

M_T = (np.pi)*(0.0047625**2)*(0.0127)*M0    # 자석의 자화벡터 값 = pi*(반지름^2)*(높이)*(자석 등급)

flag = 0    # 알고리즘 첫 시작 때만 H벡터 정규화 진행을 하기 위한 플래그변수 선언
np.set_printoptions(precision=5, suppress=True)    # 배열 변수 출력 시 소수점 아래 5자리까지만 출력되도록 설정






# 함수 선언
def callback_initial(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    for i in range(3):
        first_value[i] = data.data[i]



# offset 적용하는 함수
def offset_Setting():
    global array_Val, zero_Val, first_value, flag

    # array_Val = np.array(array_Val) - np.array(zero_Val)    # offset 적용
    
    ### 본격적인 위치추정 코드 ###
    initial_guess = first_value    # 초기 자석의 위치좌표 및 자계강도 값
    
    # 먼저 H를 정규화시켜야 함
    if(flag == 0):        # 알고리즘 첫 시작에만 정규화시키면 됨. 이 후로는 알아서 정규화 값을 기준으로 값을 추정할 것임
        initial_guess[5] = ((-1)*(initial_guess[3]*initial_guess[4]) / (initial_guess[3]+initial_guess[4]))  # Z값을 조건에 맞게 계산 후 대입 (m^2 + n^2 + p^2 = 1)
        flag = 1
    elif(flag == 1):
        initial_guess[5] = (1-(initial_guess[3]**2)-(initial_guess[4]**2))
        H_norm = np.linalg.norm(initial_guess[3:6])                    # 정규화 계산을 위한 norm값 계산
        initial_guess[3:6] = np.array(initial_guess[3:6]) / H_norm     # 초기 자계강도(H) 벡터에 대한 정규화 진행


    result_pos = least_squares(residuals, initial_guess, method='lm')    # Levenberg-Marquardt Algorithm 계산
    
    result = np.array([result_pos.x[0], result_pos.x[1], result_pos.x[2]])  # 위치 근사값만 따로 저장
    #pprint.pprint(result * 1000)                          # 위치 근사값 출력
    print(result_pos)

    # 위치추정을 위한 초기값을 이전에 구한 추정값으로 초기화
    for i in range(6):
        first_value[i] = result_pos.x[i]


    #pretty_print(array_Val)   # 가공된 센서링된 값 출력 



# 측정한 자기장 값과 계산한 자기장 값 사이의 차이를 계산하는 함수
# 여기서 오차 제곱까지 해 줄 필요는 없음. least_squares에서 알아서 계산해 줌
def residuals(init_pos):
    global array_Val, P
    differences = [0,0,0,0,0,0] # 센서 값과 계산 값 사이의 잔차 값을 저장하는 배열변수 초기화

    # 위치에 대한 잔차 값의 총합 저장
    for i in range(9):
        buffer_residual = (array_Val[i] - np.array(cal_B(init_pos, P[i]))[:3])  # 실제값과 이론값 사이의 잔차 계산
        differences[0] += buffer_residual[0]    # 각 센서들의 잔차를 각 축 성분끼리 더한다
        differences[1] += buffer_residual[1]
        differences[2] += buffer_residual[2]
        
        normalized_B = array_Val[i] / np.linalg.norm(array_Val[i])   # 측정한 자기장 값을 정규화
        differences[3:6] += normalized_B - np.array(cal_B(init_pos, P[i]))[3:6]     # 정규화된 측정 자기장 값과 예상 자기장 값 사이의 차이 계산

    #pprint.pprint(differences) # 계산한 잔차 값의 총합 출력
    return differences




# 자석의 자기밀도를 계산하는 함수 
# A: 자석의 현재 위치좌표, P: 센서의 위치좌표, H: 자석의 자계강도
def cal_B(A_and_H, P):
    global MU, M_T 
    A = [A_and_H[0], A_and_H[1], A_and_H[2]]    # 위치 값 따로 A 리스트에 저장
    H = [A_and_H[3], A_and_H[4], A_and_H[5]]    # 자계강도 값 따로 H 리스트에 저장    
    
    N_t = ((MU / MU0) * MU0 * M_T) / (4*(np.pi))    # 상수항 계산
    
    # A[0~2]: A, A[3~5]: H를 의미함
    b_x = N_t * (((3*h_dot_p(A, H, P))*(P[0]-A[0]) / (distance_3d(P,A) ** 5)) - ((H[0]) / (distance_3d(P,A) ** 3)))
    b_y = N_t * (((3*h_dot_p(A, H, P))*(P[1]-A[1]) / (distance_3d(P,A) ** 5)) - ((H[1]) / (distance_3d(P,A) ** 3))) 
    b_z = N_t * (((3*h_dot_p(A, H, P))*(P[2]-A[2]) / (distance_3d(P,A) ** 5)) - ((H[2]) / (distance_3d(P,A) ** 3)))
    
    #print([b_x, b_y, b_z])
    return [b_x, b_y, b_z, H[0], H[1], H[2]]            # 최종 자기밀도 값 반환

# 두 점 사이의 거리를 구하는 함수
def distance_3d(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)**0.5

# H dot P의 값을 계산하는 함수
def h_dot_p(A, H, P):
    return (H[0]*(P[0]-A[0])) + (H[1]*(P[1]-A[1])) + (H[2]*(P[2]-A[2]))







# 메인 함수
def main():

    global array_Val


    rospy.init_node('algorithm_pkg_node', anonymous=True)   # 해당 노드의 기본 설정
    rospy.Subscriber("initial", Float32MultiArray, callback_initial)

    # 본격적인 시뮬레이션 반복문
    while not rospy.is_shutdown():
        # print(np.array(first_value))
        


    rospy.spin()    # node 무한 반복


if __name__ == '__main__':
    main()                  # main문 무한 호출
    

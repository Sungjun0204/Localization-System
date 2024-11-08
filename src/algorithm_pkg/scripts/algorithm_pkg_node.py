#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
from std_msgs.msg import String, Float32MultiArray
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import levenberg_marquardt as LM
import pprint
import copy
import sys
import signal


## 수동 선택 변수 ##
FILTER_SELECT = 1                        # 필터 알고리즘 설정 -------- 0:Nothing | 1:MAF | 2:UKF | 
STANDARD_SELECT = 1                         # 사용하는 수식 출처 설정 ---- 1:엘버타 | 2:카네기멜론


# 전역변수 선언
zero_setting_flag = 0           # 4~5번 뒤에 offset을 작동시키기 위한 flag변수 선언*-

if(STANDARD_SELECT == 1):
    first_value = [0.0, 0.0, 0.065, 5023, 5023, 0]           # 엘버타 대학 논문 기준의 초기값(길이[m], 자계강도[H])
    # first_value = [0.0, 0.0, 0.1, 5023., 5023., 5023.]
elif(STANDARD_SELECT == 2):
    # first_value = [0.0, 0.0, 0.1, 0.0, 0.0, 0.951]
    # first_value = [0.0, 0.0, 0.2, 5000, 5000, 5000]
    first_value = [0.0, 0.0, 0.1, 4.0500, 4.0500, 4.0500]
    # first_value = [0.0, 0.0, 0.065, 10, 10, 10]       # 카네기멜론 대학 논문 기준의 초기값(길이[m], 자기밀도[mT])

full_packet = ""                # 패킷 값을 저장하는 리스트 변수
sensor_data = []                # 해체작업을 진행할 패킷 값을 저장하는 리스트 변수
packet_count = 0                       # 분할되어 들어오는 패킷을 총 10번만 받게 하기 위해 카운트를 세는 변수
is_collecting = False
lma_result = [0.0,0.0,0.0]             # 최종적으로 LMA 추정한 위치 값을 저장하는 리스트 변수 선언
result = [0.0,0.0,0.0]                 # 필터링까지 마친 최종 위치 값을 저장하는 리스트 변수 선언
filter_flag = False              # LMA추정이 완료되어 이후 필터링 작업을 시작하기 위한 플래그 변수
mns_coordi = [0.0,0.0,0.0]             # MNS의 좌표값을 저장하는 배열 변수 초기화
mns_coordi = np.array(mns_coordi, dtype=np.float64)
mns_c1 = [0.0,0.0,0.0,0.0,0.0,0.0]                  # MNS의 자기밀도 값을 저장하는 배열 변수 초기화
mns_c1 = np.array(mns_c1, dtype=np.float64)
mns_c2 = [0.0,0.0,0.0,0.0,0.0,0.0]
mns_c2 = np.array(mns_c2, dtype=np.float64)
mns_b = [0.0,0.0,0.0]
mns_b = np.array(mns_b, dtype=np.float64)
mns_h = [0.0,0.0,0.0]
mns_h = np.array(mns_h, dtype=np.float64)
normalized_mns_b = [0.0,0.0,0.0]
normalized_mns_b = np.array(normalized_mns_b, dtype=np.float64)
normalized_mns_c1 = [0.0,0.0,0.0]       # 정귀화된 MNS 자기밀도 값을 저장하는 배열 변수 초기화
normalized_mns_c1 = np.array(normalized_mns_c1, dtype=np.float64)
normalized_mns_c2 = [0.0,0.0,0.0]      
normalized_mns_c2 = np.array(normalized_mns_c2, dtype=np.float64)


## MAF 관련 변수 ##
sample_size = 20                           # MAF의 sampling data의 갯수 설정
maf_first = True                            # MAF의 첫 sampling 알고리즘과 그 이후의 알고리즘을 구분하기 위한 flag변수
data_matrix = np.zeros((3, sample_size))    # MAF를 위한 smapling data를 저장하는 3x100 행렬 선언.
index_maf = 0                               # FIFO 방식을 위한 포인터


## 센서값 저장 관련 변수 ##
array_Val = np.zeros((9, 3))     # sensor 값들의 numpy조작을 간편하게 하기 위해 옮겨 저장할 배열 변수 선언
zero_Val = np.zeros((9, 3))      # offset을 위해 배경 노이즈값을 저장할 배열 변수 선언

P = np.array([                           # hall sensor의 각 위치좌표 값 초기화(좌표 단위는 mm)
                [-118.,  118.,   0.],
                [   0.,  118.,   0.],
                [ 118.,  118.,   0.],
                [-118.,    0.,   0.],
                [   0.,    0.,   0.],      ## 5번 센서로, 중앙이 되는 센서다
                [ 118.,    0.,   0.],
                [-118., -118.,   0.],
                [   0., -118.,   0.],
                [ 118., -118.,   0.] ]) * (1e-3) # [m] 단위로 맞추기 위한 환산

# 상수 선언
MU0 = 4*(np.pi)*(1e-7)    # 진공투자율[H/m]
MU = 1.02                 # 사용하는 자석의 매질 투자율[-]
M0 = 1.320 / MU0          # 사용하는 자석의 등급에 따른 값[A/m]: 실험에서 사용하는 자석 등급은 N42 / 1.3[T]에서 [A/m]로 환산하기 위해 MU0 값을 나눔

M_T = (np.pi)*(0.0047625**2)*(0.0127)*M0   # 자석의 자화모멘트 값 = pi*(반지름^2)*(높이)*(자석 등급)

flag = 0    # 알고리즘 첫 시작 때만 H벡터 정규화 진행을 하기 위한 플래그변수 선언
np.set_printoptions(precision=5, suppress=True)    # 배열 변수 출력 시 소수점 아래 5자리까지만 출력되도록 설정

OFFSET_TIME = 1


######

MU_r_sol = 2374.339        # 솔레노이드 C1의 상대투자율
M_T_sol = 3.695            # 솔레노이드 C1의 자기모멘트 값






##################### 프로그램 강제종료를 위한 코드 ########################

def signal_handler(signal, frame): # ctrl + c -> exit program
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

######################################################################





# 함수 선언

#####################################################################################################################
#################################################### 통신 관련 함수 ####################################################

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

            # pretty_print(array_Val)                 # 패킷 처리한 값 다듬어서 출력 (확인용 코드)

            if (zero_setting_flag != OFFSET_TIME):  # flag변수가 0이면 offset 함수 호출
                zero_setting()
            else:
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
        
        # hall effect sensor 좌표계가 왼손나사 법칙 기준이므로, 오른나사 법칙 기준으로 변경
        sensor_values[0] *= -1      # 기존 x축 값 반전
        sensor_values[1] *= -1      # 기존 y축 값 반전
        temp = sensor_values[0]                 # x값과 y값 축 변경
        sensor_values[0] = sensor_values[1]
        sensor_values[1] = temp
        sensor_values = [v / 100000.0 for v in sensor_values]  # UART통신을 위해 없앴던 소수점 부활 (/100)
                                                                # hall seneor는 단위가 [uT]이므로, [mT]로 단위 통일 (/1000)
                                                                # 따라서 100,000을 나눠준다
        sensors_data.append(sensor_values)

    checksum_str = packet[packet.find('i') + 1:packet.find('Y')]
    checksum = parse_value(checksum_str)

    if raw_sum != checksum:
        return 0

    # pretty_print(sensors_data)  # 패킷에서 분리한 raw data 값 확인
    return sensors_data

#####################################################################################################################
#####################################################################################################################



# MNS의 좌표 값을 별도의 변수에 저장하는 callback 함수
def scara_coordi_callback(data):
    global mns_coordi

    # 위치 값은 mm 단위로 받고 있음
    mns_coordi[:3] = np.array(data.data[:3])


# C-Mag MNS가 생성하는 자기밀도 값을 받아오는 callback 함수
def c_mag_b_callback(data):
    global mns_c1, mns_c2, MU0, STANDARD_SELECT, mns_b, normalized_mns_b, mns_h

    if(STANDARD_SELECT == 2):  # 카네기멜론 대학 기준일 때
        # C1,C2 위치벡터와 자기모멘트 값 배분
        # mns_c1 = np.array(copy.deepcopy(data.data[:6]))
        # mns_c2 = np.array(copy.deepcopy(data.data[6:]))
        # mns_b = data.data
        # mns_b = np.array([0.02275, -0.00146, 0.2, 2954.06852, 60361.78393, -1475.85602])

        # C-Core 좌표계에서 hall sensro 좌표계로 변경
        mns_b = data.data
        tf_core2sensor = np.array([[0,0,-1],[-1,0,0],[0,1,0]])
        mns_b = np.dot(tf_core2sensor, np.array(mns_b))

        # mns_c1[3:] = np.dot(tf_core2sensor, np.array(mns_c1[3:]))
        # mns_c2[3:] = np.dot(tf_core2sensor, np.array(mns_c2[3:]))

    elif(STANDARD_SELECT == 1):  # 엘버타 논문 기준

        # MNS 1개로
        # B 값 받아옴
        mns_b[0] = data.data[0]
        mns_b[1] = data.data[1]
        mns_b[2] = data.data[2]
        tf_core2sensor = np.array([[0,0,-1],[-1,0,0],[0,1,0]])
        mns_b = np.dot(tf_core2sensor, np.array(mns_b))
        
        # 자계강도로 환산
        mns_h[0] = mns_b[0]
        mns_h[1] = mns_b[1]
        mns_h[2] = mns_b[2]
        norm_mns_b = np.linalg.norm(mns_h)  

        # C1,C2 위치벡터와 자계강도(이미 정규화 됨) 값 배분
        # mns_c1 = np.array(data.data[:6])
        # mns_c2 = np.array(data.data[6:])

        # 정규화 진행해서 별도 배열에 저장
        if norm_mns_b != 0:
            normalized_mns_b = mns_b / norm_mns_b
        else:
            # MNS의 자기장 성분의 norm이 0일 때의 처리
            normalized_mns_b = np.zeros_like(normalized_mns_b)





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

    zero_setting_flag += 1                   # 이후 1로 설정하여 함수가 호출되지 않도록 초기화
                                            # 이렇게 안 하면 센서 측정 값이 계속 0으로만 출력됨


# offset 적용하는 함수
def offset_Setting():
    global array_Val, zero_Val, first_value, flag, lma_result, filter_flag
    global result, sample_size, maf_first, data_matrix, index_maf, FILTER_SELECT, STANDARD_SELECT
    
    array_Val = np.array(array_Val) - np.array(zero_Val)    # offset 적용 (평소에는 zero_val = 0이므로 발동 안 됨)
    
    mean_vector = np.mean(array_Val, axis=0)    # 9개의 센서 값에 대한 평균
    norm_vector = np.linalg.norm(mean_vector)   # 계산한 평균 벡터의 norm을 계산

    # print(array_Val)
    
    ##########################################
    #### LMA를 이용해 차석의 위치를 추정하는 코드 ####
    ##########################################
    
    # 평균 norm 값이 0.005 이상이면(=자석이 센서 배열에서부터 20cm 이내로 위치하면)
    if(norm_vector > 0.005):
        ### 본격적인 위치추정 코드 ###
        initial_guess = first_value    # 초기 자석의 위치좌표 및 자계강도 값
        
        # 먼저 초기값 H(내가 설정하는 값)를 정규화시켜야 함
        if(STANDARD_SELECT == 1): 
            if(flag == 0):        # 알고리즘 첫 시작에만 정규화시키면 됨. 이 후로는 알아서 정규화 값을 기준으로 값을 추정할 것임
                initial_guess[5] = ((-1)*(initial_guess[3]*initial_guess[4]) / (initial_guess[3]+initial_guess[4]))  # Z값을 조건에 맞게 계산 후 대입 (m^2 + n^2 + p^2 = 1)
                flag = 1          # 이후 flag 값을 1로 설정하여 더 이상 중복으로 정규화 계산하지 않도록 처리
            elif(flag == 1):
                initial_guess[5] = (1-(initial_guess[3]**2)-(initial_guess[4]**2))
                H_norm = np.linalg.norm(initial_guess[3:6])                    # 정규화 계산을 위한 norm값 계산
                initial_guess[3:6] = np.array(initial_guess[3:6]) / H_norm     # 초기 자계강도(H) 벡터에 대한 정규화 진행
        
        ## LMA 연산
        # 엘버타 기준 수식 사용
        if(STANDARD_SELECT == 1):
            result_pos = least_squares(residuals, initial_guess, method='lm')    # Levenberg-Marquardt Algorithm 계산
        
        # 카네기멜론 기준 수식 사용
        elif(STANDARD_SELECT == 2):
            result_pos = least_squares(residuals2, initial_guess, method='lm')    # Levenberg-Marquardt Algorithm 계산


        # lma_result = [result_pos.x[0], result_pos.x[1], result_pos.x[2]]  # 위치 근사값만 따로 저장
        lma_result = result_pos.x[:6]
        
        # if(lma_result[0] >= 0.2 or lma_result[1] >= 0.2 or lma_result[0] <= -0.2 or lma_result[1] <= -0.2):
        #     lma_result = [0.0, 0.0, 0.1, 10, 10, 10]

        # print("... Measuring ...")
        
    
        
        
        
        ######################################
        #### LMA 추정 이후 필터를 적용하는 코드 ####
        ######################################

        #### filter select 0: Any Filtering ####
         # 위치추정을 위한 초기값을 이전에 구한 추정값으로 초기화
        if(FILTER_SELECT == 0):
            for i in range(6):
                first_value[i] = result_pos.x[i]

            for i in range(3):
                result[i] = result_pos.x[i]


        
        #### filter select 1: MAF ####
        if(FILTER_SELECT == 1):
            for i in range(6):
                first_value[i] = result_pos.x[i]
            
            # MAF의 sampling 설정
            if (maf_first == True):
                data_matrix[0, index_maf] = first_value[0]   # x값 샘플링
                data_matrix[1, index_maf] = first_value[1]   # y값 샘플링
                data_matrix[2, index_maf] = first_value[2]   # z값 샘플링
                
                if(index_maf == sample_size-1):
                    filter_flag = True      # sampling data가 100개가 다 되면 MAF 필터링 시작.
                    maf_first = False       # 동시에 첫 sampling 알고리즘이 아닌 다른 필터링 알고리즘 사용 시작
                else:
                    index_maf += 1          # 첫 sampling data가 100개가 아직 안 모였다면 인덱스 +1

            elif(maf_first == False):
                # data_matrix[:, 1:] = data_matrix[:, :-1]    # sampling data를 왼쪽으로 이동
                # data_matrix[:, -1] = first_value[:3]        # 최신 data 추가
                new_elements = np.array(first_value[:3])  # 각 행에 추가할 새 원소
                data_matrix = np.hstack((data_matrix[:, 1:], new_elements[:, np.newaxis]))


            if(filter_flag == True):            # filter_flag가 켜지면 해당 함수 실행
                for i in range(3):
                    result[i] = maf_func(data_matrix[i])
                # print(result)
        


        ##############################
        #### filter select 2: UKF ####
        ##############################
        elif(FILTER_SELECT == 2):
            # 상태 차원과 측정 차원 설정
            dim_x = 6  # 상태변수 차원 (x, y, z 위치, bx, by, bz 자기밀도)
            dim_z = 6  # 측정변수 차원
            dt = 0.1

            # 시그마 포인트 생성을 위한 매개변수
            points = MerweScaledSigmaPoints(n=dim_x, alpha=0.01, beta=2., kappa=0.5)
            # points = MerweScaledSigmaPoints(n=dim_x, alpha=0.001, beta=2., kappa=0)

            # UKF 초기화
            ukf = UKF(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, points=points)
            ukf.x = first_value[:6]      # 초기 상태 추정치를 마찬가지로 first_value 값을 가져옴
            ukf.P *= np.eye(dim_x) * 0.2
            # ukf.P *= np.cov(first_value - lma_result)          # 초기 상태 추정치의 불확실성(초기값에서 측정값 뺀 값)
            


            # ukf.R = np.eye(dim_z) * 1.0  # 관측 노이즈: 측정 시스템에 노이즈가 많을수록 큰 값을 사용
            # ukf.Q = np.eye(dim_x) * 0.4  # 프로세스 노이즈: 시스템의 변화가 빠르고 복잡할수록 큰 값을 사용
            
            # 엘버타 논문 기준
            if(STANDARD_SELECT == 1):
                # ukf.R = np.diag([900, 900, 900, 20, 20, 20])
                # ukf.Q = np.diag([0.05, 0.05, 0.05, 200, 200, 200])
                # ukf.R = np.diag([1.41, 1.41, 1.41, 29, 29, 29])
                # ukf.Q = np.diag([11, 11, 11, 50.5, 50.5, 50.5])
                ukf.R = np.eye(dim_z) * 0.1
                ukf.Q = np.eye(dim_x) * 10
                # ukf.R = np.diag([10, 10, 10, 10, 10, 10])
                # ukf.Q = np.diag([0.5, 0.5, 0.5, 10, 10, 10])

        
            # 카네기멜론 논문 기준
            elif(STANDARD_SELECT == 2):
                # ukf.R = np.diag([0.03, 0.03, 0.03, 0.05, 0.05, 0.05])
                # ukf.Q = np.diag([0.5, 0.5, 0.5, 0.3, 0.3, 0.3])
                # ukf.R = np.diag([0.01, 0.01, 0.01, 0.5, 0.5, 0.5])
                # ukf.Q = np.diag([0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
                # ukf.R = np.diag([0.3, 0.3, 0.3, 0.5, 0.5, 0.5])
                # ukf.Q = np.diag([10, 10, 10, 0.5, 0.5, 0.5])
                ukf.R = np.diag([1e-6, 1e-6, 1e-6, 1e-2, 1e-2, 1e-2])
                ukf.Q = np.diag([1e-6, 1e-6, 1e-6, 1e-2, 1e-2, 1e-2])
                # ukf.R = np.eye(dim_z) * 10
                # ukf.Q = np.eye(dim_x) * 200

 


            # 예제 측정 업데이트
            z = lma_result[:6]               # 측정 값으로 LMA 최종 추정값을 가져옴
            ukf.predict()
            ukf.update(z)

            # 위치추정을 위한 초기값을 이전에 구한 추정값으로 초기화
            for i in range(6):
                first_value[i] = ukf.x[i]

            result = ukf.x[:6]                      # 진짜 최종 위치 값만 따로 저장
            # print(ukf.x)                # 값 확인
            
        



        
    # 평균 norm 값이 0.005 이하면(=자석이 센서 배열에서부터 20cm 이상 떨어져 있으면)
    else:
        print("!! Out of Workspace !!") # 해당 텍스트만 출력하고 알고리즘 작동은 일절 없음
    
    






#### 측정한 자기장 값과 계산한 자기장 값 사이의 차이를 계산하는 함수 ####
# 여기서 오차 제곱까지 해 줄 필요는 없음. least_squares에서 알아서 계산해 줌
def residuals(init_pos):
    global array_Val, P, MU0, mns_coordi, mns_c1, mns_c2, mns_b, normalized_mns_b
    
    # num = 0.0267
    num = 1

    differences = [] # 센서 값과 계산 값 사이의 잔차 값을 저장하는 배열변수 초기화

    # mns_value = []
    # mns_value.extend(mns_coordi * 1e-3) # 알고리즘에는 [m]단위로 환산해서 들어감
    # mns_value.extend(normalized_mns_b)
    mns_value = np.array([-0.0005, -0.00179, 0.12377, 0.00644, -0.03073, -0.78245])

    #  # 제약 조건 설정 (예: init_pos의 각 요소가 특정 범위를 넘지 않도록 함)
    lower_bound = np.array([-2, -2, 2, -1, -1, -1])  # 최소값 설정
    upper_bound = np.array([2, 2, 2, 1, 1, 1])  # 최대값 설정

    penalty = 1e6  # 벌점 항목 (제약 조건 위반 시 추가)


    # ### 방법1: 솔레노이드 2개 별도로
    # mns_c1_copy = mns_c1.copy()
    # mns_c2_copy = mns_c2.copy()

    # mns_coordi_m = mns_coordi * 1e-3

    # # # 각 C1, C2의 위치는 SCARA의 End Effect가 움직인 만큼 똑같이 움직인다 가정
    # mns_c1_copy[:3] = mns_c1[:3] + mns_coordi_m
    # mns_c2_copy[:3] = mns_c2[:3] + mns_coordi_m


    # # 벌점 계산: 범위 벗어난 경우 해당 요소에 penalty를 추가
    constraint_violations = np.logical_or(init_pos < lower_bound, init_pos > upper_bound)
    penalty_value = penalty * constraint_violations  # 제약을 벗어난 경우 벌점 부여
    differences.extend(penalty_value)
    

    print(init_pos)
    # print(mns_value)

    # 위치에 대한 잔차 값의 총합 저장
    for i in range(9):
        ## 위치값에 대한 잔차 계산
        # calculated_B = np.array(cal_B2(mns_value, P[i])[:3]) + np.array(cal_B(init_pos, P[i])[:3])
        calculated_B = cal_B(init_pos, P[i])[:3]
        buffer_residual = array_Val[i][:3]*num - calculated_B[:3]
        differences.extend(buffer_residual)

        ## 자계강도 값에 대한 잔차 계산
        normalized_B = (array_Val[i]) / np.linalg.norm(array_Val[i])
        # calculated_B_normalized = np.array(cal_B2(mns_value, P[i])[3:6]) + np.array(cal_B(init_pos, P[i])[3:6])
        calculated_B_normalized = cal_B(init_pos, P[i])[3:6]
        differences.extend(normalized_B[:3] - calculated_B_normalized[:3])


    return differences    # 최종적으로 6x9=54개의 잔차값이 저장된 리스트를 반환


############ 엘버타 대학 논문 기준 ############
#### 자석의 자기밀도를 계산하는 함수 ####
# A: 자석의 현재 위치좌표, P: 센서의 위치좌표, H: 자석의 자계강도
def cal_B(A_and_H, P):
    global MU, M_T, MU0
    global MU_r_sol, M_T_sol
    A = A_and_H[:3]    # 자석 위치 값 따로 A 리스트에 저장
    H = A_and_H[3:]    # 자계강도 값 따로 H 리스트에 저장    
    
    N_t = ((MU / MU0) * MU0 * M_T) / (4*(np.pi))    # 상수항 계산
    # N_t = (MU_r_sol * MU0 * M_T_sol) / (4*(np.pi))    # 솔레노이드 C1 기준 상수항 계산
    
    # 거리 계산
    R = P - A
    Rn = np.linalg.norm(R)
    if Rn == 0:
        return np.zeros_like(6)

    h_dot_p = np.dot(H,R)
    
    # A[0~2]: A, A[3~5]: H를 의미함
    b = N_t * ((3 * h_dot_p * R / (Rn ** 5)) - (H / (Rn ** 3)))

    return [b[0], b[1], b[2], H[0], H[1], H[2]]            # 최종 자기밀도 값 반환 (자계강도 값은 앞서 계산한 값 그대로 저장)
    # return np.concatenate((b, H))  # 자기밀도 값과 자계강도 값을 같은 배열로 합쳐 반환
    # return b


def cal_B2(A_and_H, P):
    global MU, M_T, MU0
    global MU_r_sol, M_T_sol
    A = A_and_H[:3]    # 자석 위치 값 따로 A 리스트에 저장
    H = A_and_H[3:]    # 자계강도 값 따로 H 리스트에 저장    
    
    # N_t = ((MU / MU0) * MU0 * M_T) / (4*(np.pi))    # 상수항 계산
    N_t = (MU_r_sol * MU0 * M_T_sol) / (4*(np.pi))    # 솔레노이드 C1 기준 상수항 계산
    
    # 거리 계산
    R = P - A
    Rn = np.linalg.norm(R)
    if Rn == 0:
        return np.zeros_like(6)

    h_dot_p = np.dot(H,R)
    
    # A[0~2]: A, A[3~5]: H를 의미함
    b = N_t * ((3 * h_dot_p * R / (Rn ** 5)) - (H / (Rn ** 3)))

    return [b[0], b[1], b[2], H[0], H[1], H[2]]







############ 카네기멜론 대학 논문 기준 ############
def residuals2(init_pos):
    global array_Val, P, mns_coordi, mns_c1, mns_c2, mns_b
    differences = []                        # (센서 값)과 (계산 값) 사이의 잔차 값을 저장하는 리스트변수 초기화

     # 제약 조건 설정 (예: init_pos의 각 요소가 특정 범위를 넘지 않도록 함)
    lower_bound = np.array([-0.2, -0.2, -0.2, -1e5, -1e5, -1e5])  # 최소값 설정
    upper_bound = np.array([0.2, 0.2, 0.2, 1e5, 1e5, 1e5])  # 최대값 설정

    penalty = 1e6  # 벌점 항목 (제약 조건 위반 시 추가)
    max_moment_norm = 7.013  # 자기모멘트 벡터의 최대 norm 값 제한

    # 원래방법
    mns_value = []
    mns_value.extend(mns_coordi * 1e-3) # 알고리즘에는 [m]단위로 환산해서 들어감
    mns_value.extend(mns_b)
    
    # mns_coordi_m = mns_coordi * 1e-3

    # ### 방법1: 솔레노이드 2개 별도로
    # mns_c1_copy = mns_c1.copy()
    # mns_c2_copy = mns_c2.copy()

    # 각 C1, C2의 위치는 SCARA의 End Effect가 움직인 만큼 똑같이 움직인다 가정
    # mns_c1_copy[:3] = mns_c1[:3] + mns_coordi_m
    # mns_c2_copy[:3] = mns_c2[:3] + mns_coordi_m

    ### 방법2: 솔레노이드 1개로
    # mns_b_copy = mns_b.copy()
    # mns_b_copy[:3] = mns_b[:3] + mns_coordi_m
    # mns_value.extend(x * 1e-3 for x in mns_coordi) # 알고리즘에는 [m]단위로 환산해서 들어감
    # mns_value.extend(mns_b) # 자계강도 값에서 자기밀도 값으로 환산해서 들어감

    val = array_Val.reshape(3,3,3)           # 센서 값을 3x3 형태로 다시 저장(for 계산 용이)
    hh = 0.118                               # 센서 간 거리의 제곱의 역수 계산(단위 m)

    print(init_pos)
    # print(mns_value)

    # 벌점 계산: 범위 벗어난 경우 해당 요소에 penalty를 추가
    constraint_violations = np.logical_or(init_pos < lower_bound, init_pos > upper_bound)
    penalty_value = penalty * constraint_violations.sum()  # 제약 위반 총합을 벌점으로 사용
    # 벌점을 differences 배열에 추가하되, 크기를 일정하게 유지
    differences.append(penalty_value)

    # 추가 제약 조건: 자기 모멘트 벡터의 norm 제한
    moment_vector = init_pos[3:6]  # 위치 벡터 뒤에 위치한 자기모멘트 벡터
    moment_norm = np.linalg.norm(moment_vector)

    if moment_norm > max_moment_norm:
        # norm이 최대값을 초과할 때 벌점을 differences에 추가하되 크기 유지
        norm_penalty = penalty * (moment_norm - max_moment_norm)
        differences.append(norm_penalty)
    else:
        differences.append(0)  # 조건을 만족할 경우 0을 추가하여 크기 유지


    # 각 센서마다 K_ij 값 계산 (논문의 식 11) -> 총 9개의 K_ij값이 계산됨
    # 센서 배열이 3x3이므로, 0보다 작거나 2보다 크면 해당 센서 위치의 값은 0으로 처리
    for i in range(3):
        for j in range(3):
            param = np.array([0.,0.,0.,0.,0.], dtype=np.float64)
            param[0]=val[i][j][2]*(0.225) if j-1 < 0 else val[i][j-1][2]
            param[1]=val[i][j][2]*(0.225) if j+1 > 2 else val[i][j+1][2]
            param[2]=val[i][j][2]*(0.225) if i-1 < 0 else val[i-1][j][2]
            param[3]=val[i][j][2]*(0.225) if i+1 > 2 else val[i+1][j][2]
            # param[0]=0 if j-1 < 0 else val[i][j-1][2]
            # param[1]=0 if j+1 > 2 else val[i][j+1][2]
            # param[2]=0 if i-1 < 0 else val[i-1][j][2]
            # param[3]=0 if i+1 > 2 else val[i+1][j][2]
            param[4]=(-4)*(val[i][j][2])
            k_ij = (-1) * (sum(param) / (hh**2))  # K_ij 배열에 하나씩 추가
            
            # 중앙 센서 (1,1)의 경우 -> five-point stencil 적용
            # if(i==1 and j==1):
            #     k_ij = (-1) * (val[i][j-1][2] + val[i][j+1][2] + val[i-1][j][2] + val[i+1][j][2] - 4*val[i][j][2]) / (hh**2)

            # # 모서리 센서 (0,0),(0,2),(2,0),(2,2)의 경우 -> three-point stencil을 4차 항까지 적용
            # elif (i == 0 and j == 0) or (i == 0 and j == 2) or (i == 2 and j == 0) or (i == 2 and j == 2):
            #     term1 = 0; term2 = 0
            #     for k in range(1,3):
            #         term_arr = [0,0,0,0,0]
            #         term_arr[0]=0 if j-k < 0 else val[i][j-k][2]
            #         term_arr[1]=0 if j+k > 2 else val[i][j+k][2]
            #         term_arr[2]=0 if i-k < 0 else val[i-k][j][2]
            #         term_arr[3]=0 if i+k > 2 else val[i+k][j][2]
            #         term_arr[4]=(-2)*(val[i][j][2])

            #         if(k==1):    term1 = (-1) * sum(term_arr) / (hh**2)
            #         elif(k==2):  term2 = (-1) * sum(term_arr) * 12
                
            #     k_ij = term1 - term2

            # # 경계면 센서 (0,1),(1,0),(1,2),(2,1)의 경우 -> three-point stencil을 적용
            # elif (i == 0 and j == 1) or (i == 1 and j == 0) or (i == 1 and j == 2) or (i == 2 and j == 1):
            #     if(j==1):    # j==1이면 세로로 적용
            #         k_ij = (-1) * (val[i][j-1][2] + val[i][j+1][2] -2*val[i][j][2]) / (hh**2)
                
            #     elif(i==1):  # i==1이면 가로로 적용
            #         k_ij = (-1) * (val[i-1][j][2] + val[i+1][j][2] -2*val[i][j][2]) / (hh**2)

            # else: k_ij = 0
            

            # buffer_residual = k_ij - cal_BB(mns_c1_copy, P[(i*3)+j]) - cal_BB(init_pos, P[(i*3)+j])  # 실제값과 이론값 사이의 잔차 계산
            buffer_residual = k_ij - cal_BB(init_pos, P[(i*3)+j])
            # buffer_residual = k_ij - cal_BB(mns_value, P[(i*3)+j]) - cal_BB(init_pos, P[(i*3)+j])
            differences.append(buffer_residual)    # 각 센서들의 잔차 값을 differences 배열에 1차원으로 삽입
    
    # pprint.pprint(differences) # 계산한 잔차 값의 총합 출력
    # print("----------------------")
    return differences






# 자석의 자기밀도를 계산하는 함수 (논문의 식 15)
# A: 자석의 현재 위치좌표, P: 센서의 위치좌표, H: 자석의 자계강도
def cal_BB(A_and_H, P):
    global MU0
    A = A_and_H[:3]   # 자석의 위치 값 따로 A 리스트에 저장
    M = A_and_H[3:]   # 자석의 자기모멘트 벡터 값 따로 H 리스트에 저장
    R = np.array(P-A)                          # ij번째 센서와 자석 사이의 거리벡터
    Rn = np.linalg.norm(R)                     # ij번째 센서와 자석 사이의 거리(norm1)
    if Rn == 0:
        Rn = np.zeros_like(6)

    const = MU0 / (4*np.pi)     # 상수항 계산
    b1 = (9*M[2]) / (Rn ** 5)
    b2 = (45*(R[2])*(np.dot(M,R)+(M[2]*R[2]))) / (Rn ** 7)
    b3 = (105*(R[2]**3)*(np.dot(M,R))) / (Rn ** 9)

    return const*(b1-b2+b3)







#### Kalman Filter의 상태 천이 행렬을 반환하는 함수 ####
def fx(state, dt):
    F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    return state

#### Kalman Filter에서, 측정 함수로 좌표변환하는 함수 ####
def hx(state):
    return state

#### MAF 함수 ####
def maf_func(samples):
    
    # 데이터 생성
    data = np.array(samples)

    # MAF 필터링
    filtered_data = data.sum() / len(samples)

    return filtered_data






################################### 메인 함수 #######################################
def main():

    global array_Val, result, P, mns_coordi


    rospy.init_node('algorithm_pkg_node', anonymous=True)   # 해당 노드 기본 설정
    
    #### 메세지 발행 설정 구간 ####
    pub_mag    = rospy.Publisher('visualization_marker', Marker, queue_size=10)   # 최종 추정한 자석의 위치좌표
    pub_mns    = rospy.Publisher('mns_marker', Marker, queue_size=10)      # MNS의 위치좌표
    pub_sensor = rospy.Publisher('sensors_marker', Marker, queue_size=10)  # 센서 위치 좌표

    #### 메세지 구독 설정 구간 ####
    rospy.Subscriber('scara_coordi', Float32MultiArray, scara_coordi_callback) # /scara_coordi를 구독하고 scara_coordi_callback 함수 호출
    rospy.Subscriber('c_mag_b', Float32MultiArray, c_mag_b_callback) # /c_mag_b를 구독하고 c_mag_b_callback 함수 호출
    rospy.Subscriber('read', String, seperating_Packet)   # /read를 구독하고 seperating_Packet 함수 호출: 패킷 처리 함수
    rospy.Subscriber('Is_offset', String, callback_offset)  # /Is_offset을 구독하고 callback_offset 함수 호출

    rate = rospy.Rate(10)  # 100Hz


    #### 메인 반복문 ####
    while (not rospy.is_shutdown()):
        # 추정된 자석의 위치 마커 생성
        marker = Marker()
        marker.header = Header(frame_id="map", stamp=rospy.Time.now())
        marker.ns = "my_namespace"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        if(STANDARD_SELECT == 1):
            marker.pose.position.x = result[0] * 100
            marker.pose.position.y = result[1] * 100
        elif(STANDARD_SELECT == 2):
            marker.pose.position.x = result[0] * 1000
            marker.pose.position.y = result[1] * 1000
        marker.pose.position.z = 0#result[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 5  # 마커의 크기
        marker.scale.y = 5
        marker.scale.z = 5
        marker.color.a = 1.0  # 마커의 투명도
        marker.color.r = 1.0  # 마커의 색상
        marker.color.g = 0.0
        marker.color.b = 0.0
        pub_mag.publish(marker)

        # 넘겨받은 MNS의 위치 마커 생성
        marker2 = Marker()
        marker2.header = Header(frame_id="map", stamp=rospy.Time.now())
        marker2.ns = "my_namespace"
        marker2.id = 0
        marker2.type = Marker.SPHERE
        marker2.action = Marker.ADD
        marker2.pose.position.x = mns_coordi[0]
        marker2.pose.position.y = mns_coordi[1]
        marker2.pose.position.z = 0#result[2]
        marker2.pose.orientation.x = 0.0
        marker2.pose.orientation.y = 0.0
        marker2.pose.orientation.z = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.scale.x = 5  # 마커의 크기
        marker2.scale.y = 5
        marker2.scale.z = 5
        marker2.color.a = 1.0  # 마커의 투명도
        marker2.color.r = 0.5  # 마커의 색상
        marker2.color.g = 0.3
        marker2.color.b = 0.4
        pub_mns.publish(marker2)

        # 각 센서 위치에 대한 마커 발행
        for i, point in enumerate(P * 1000):
            marker3 = Marker()
            marker3.header = Header(frame_id="map", stamp=rospy.Time.now())
            marker3.ns = "my_namespace"
            marker3.id = i
            marker3.type = Marker.CUBE
            marker3.action = Marker.ADD
            marker3.pose.position.x = point[0]
            marker3.pose.position.y = point[1]
            marker3.pose.position.z = point[2]-5
            marker3.pose.orientation.x = 0.0
            marker3.pose.orientation.y = 0.0
            marker3.pose.orientation.z = 0.0
            marker3.pose.orientation.w = 1.0
            marker3.scale.x = 5  # 마커 크기
            marker3.scale.y = 5
            marker3.scale.z = 5

            # 마커 색상 설정
            if i <= 6:  # 0~6번 마커는 녹색
                marker3.color.r = 0.6
                marker3.color.g = 1.0
                marker3.color.b = 0.2
            else:  # 7~8번 마커는 파란색
                marker3.color.r = 0.0
                marker3.color.g = 0.0
                marker3.color.b = 1.0

            marker3.color.a = 1.0  # 마커 투명도
            pub_sensor.publish(marker3)


        rate.sleep()

    rospy.spin()    # node 무한 반복


if __name__ == '__main__':
    main()                  # main문 무한 호출
    
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
import sys
import signal


## 수동 선택 변수 ##
FILTER_SELECT = 2                           # 필터 알고리즘 설정 ---------1:MAF | 2:UKF | 


# 전역변수 선언
zero_setting_flag = 0           # 4~5번 뒤에 offset을 작동시키기 위한 flag변수 선언*-

first_value = [0.0, 0.0, 0.065, 4.509, 4.509, 4.509]       # 카네기멜론 대학 논문 기준의 초기값(길이[m], 자기밀도[mT])

full_packet = ""                # 패킷 값을 저장하는 리스트 변수
sensor_data = []                # 해체작업을 진행할 패킷 값을 저장하는 리스트 변수
packet_count = 0                       # 분할되어 들어오는 패킷을 총 10번만 받게 하기 위해 카운트를 세는 변수
is_collecting = False
lma_result = [0,0,0]             # 최종적으로 LMA 추정한 위치 값을 저장하는 리스트 변수 선언
result = [0,0,0]                 # 필터링까지 마친 최종 위치 값을 저장하는 리스트 변수 선언
filter_flag = False              # LMA추정이 완료되어 이후 필터링 작업을 시작하기 위한 플래그 변수
mns_coordi = [0,0,0]             # MNS의 좌표값을 저장하는 배열 변수 초기화
mns_coordi = np.array(mns_coordi)
mns_b = [0,0,0]                  # MNS의 자기밀도 값을 저장하는 배열 변수 초기화
mns_b = np.array(mns_b)
normalized_mns_b = [0,0,0]       # 정귀화된 MNS 자기밀도 값을 저장하는 배열 변수 초기화
normalized_mns_b = np.array(normalized_mns_b)
previous_z = [0,0,0,0,0,0]       # UKF에서 사용될 추가 측정 값(= 이전 스텝에서 구한 최종 추정값)
previous_z = np.array(previous_z)


## MAF 관련 변수 ##
sample_size = 20                           # MAF의 sampling data의 갯수 설정
maf_first = True                            # MAF의 첫 sampling 알고리즘과 그 이후의 알고리즘을 구분하기 위한 flag변수
data_matrix = np.zeros((3, sample_size))    # MAF를 위한 smapling data를 저장하는 3x100 행렬 선언.
index_maf = 0                               # FIFO 방식을 위한 포인터


## 센서값 저장 관련 변수 ##
array_Val = np.zeros((9, 3))     # sensor 값들의 numpy조작을 간편하게 하기 위해 옮겨 저장할 배열 변수 선언
zero_Val = np.zeros((9, 3))      # offset을 위해 배경 노이즈값을 저장할 배열 변수 선언

P = np.array([                           # hall sensor의 각 위치좌표 값 초기화(좌표 단위는 mm)
                [-118,  118,   0],
                [   0,  118,   0],
                [ 118,  118,   0],
                [-118,    0,   0],
                [   0,    0,   0],      ## 5번 센서로, 중앙이 되는 센서다
                [ 118,    0,   0],
                [-118, -118,   0],
                [   0, -118,   0],
                [ 118, -118,   0] ]) * (1e-3) # [m] 단위로 맞추기 위한 환산

# 상수 선언
MU0 = 4*(np.pi)*(1e-7)    # 진공투자율[H/m]
MU = 1.02                 # 사용하는 자석의 매질 투자율[-]
M0 = 1.320 / MU0          # 사용하는 자석의 등급에 따른 값[A/m]: 실험에서 사용하는 자석 등급은 N42 / 1.3[T]에서 [A/m]로 환산하기 위해 MU0 값을 나눔

M_T = (np.pi)*(0.0047625**2)*(0.0127)*M0   # 자석의 자화벡터 값 = pi*(반지름^2)*(높이)*(자석 등급)

flag = 0    # 알고리즘 첫 시작 때만 H벡터 정규화 진행을 하기 위한 플래그변수 선언
np.set_printoptions(precision=5, suppress=True)    # 배열 변수 출력 시 소수점 아래 5자리까지만 출력되도록 설정

OFFSET_TIME = 1




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
        sensor_values[0] *= -1
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

    # mm 단위로 들어오므로 추후 사용할 때 가공할 필요가 있음
    mns_coordi[0] = data.data[0] 
    mns_coordi[1] = data.data[1] 
    mns_coordi[2] = data.data[2]


# C-Mag MNS가 생성하는 자기밀도 값을 받아오는 callback 함수
def c_mag_b_callback(data):
    global mns_b, MU0

    # B[mT] 값으로 구독하므로 그대로 옮겨 저장
    mns_b[0] = data.data[0]
    mns_b[1] = data.data[1]
    mns_b[2] = data.data[2] 





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
    global array_Val, zero_Val, first_value, flag, lma_result, filter_flag, previous_z
    global result, sample_size, maf_first, data_matrix, index_maf, FILTER_SELECT
    
    array_Val = np.array(array_Val) - np.array(zero_Val)    # offset 적용
    
    mean_vector = np.mean(array_Val, axis=0)    # 9개의 센서 값에 대한 평균
    norm_vector = np.linalg.norm(mean_vector)   # 계산한 평균 벡터의 norm을 계산
    
    ##########################################
    #### LMA를 이용해 차석의 위치를 추정하는 코드 ####
    ##########################################
    
    # 평균 norm 값이 0.005 이상이면(=자석이 센서 배열에서부터 20cm 이내로 위치하면)
    if(norm_vector > 0.005):
        ### 본격적인 위치추정 코드 ###
        initial_guess = first_value    # 초기 자석의 위치좌표 및 자계강도 값

        
        ## LMA 연산
        # 카네기멜론 기준 수식 사용
        result_pos = least_squares(residuals2, initial_guess, method='lm')    # Levenberg-Marquardt Algorithm 계산

        lma_result = result_pos.x[:6] # 추정 값의 결과 값 6개만 따로 저장 (그냥 다 저장하면 별의 별 파라미터들이 저장됨)
        

        # print("... Measuring ...")
        
        # 위치추정을 위한 초기값을 이전에 구한 추정값으로 초기화
        if(FILTER_SELECT == 0):
            for i in range(6):
                first_value[i] = result_pos.x[i]
            
            for i in range(3):
                result[i] = result_pos.x[i]
        
        
        
        ######################################
        #### LMA 추정 이후 필터를 적용하는 코드 ####
        ######################################

        #### filter select 1: Any Filtering ####
        if(FILTER_SELECT == 1):
            for i in range(6):
                first_value[i] = result_pos.x[i]

            # result = first_value[:3]   # rviz에 띄위기 위해 위치값만 따로 저장
        
        #### filter select 1: MAF ####
        if(FILTER_SELECT == 1):
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
            points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2., kappa=0.5)
            # points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2., kappa=1)

            # UKF 초기화
            ukf = UKF(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, points=points)
            ukf.x = first_value[:6]      # 초기 상태 추정치를 마찬가지로 first_value 값을 가져옴
            # ukf.P *= np.cov(first_value - lma_result)     # 초기 상태 추정치의 불확실성(초기값에서 측정값 뺀 값)
            ukf.P *= 0.2
            # ukf.R = np.eye(dim_z) * 1.0  # 관측 노이즈: 측정 시스템에 노이즈가 많을수록 큰 값을 사용
            # ukf.Q = np.eye(dim_x) * 0.4  # 프로세스 노이즈: 시스템의 변화가 빠르고 복잡할수록 큰 값을 사용
        
            # 카네기멜론 논문 기준
            # ukf.R = np.diag([0.01, 0.01, 0.01, 0.5, 0.5, 0.5])
            # ukf.Q = np.diag([0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
            ukf.R = np.diag([0.01, 0.01, 0.01, 1.1, 1.1, 1.1])
            ukf.Q = np.diag([0.1, 0.1, 0.1, 0.4, 0.4, 0.4])


            # 예제 측정 업데이트
            # z = lma_result[:6]               # 측정 값으로 LMA 최종 추정값을 가져옴
            z = previous_z
            ukf.predict()
            ukf.update(z)

            # 위치추정을 위한 초기값을 이전에 구한 추정값으로 초기화
            for i in range(6):
                first_value[i] = ukf.x[i]
                previous_z[i] = ukf.x[i]

            result = ukf.x[:6]                      # 진짜 최종 위치 값만 따로 저장
            # print(ukf.x)                # 값 확인
            
        



        
    # 평균 norm 값이 0.005 이하면(=자석이 센서 배열에서부터 20cm 이상 떨어져 있으면)
    else:
        print("!! Out of Workspace !!") # 해당 텍스트만 출력하고 알고리즘 작동은 일절 없음
    
    





############ 카네기멜론 대학 논문 기준 ############
def residuals2(init_pos):
    global array_Val, P, first_value, mns_coordi, mns_b
    differences = []                        # (센서 값)과 (계산 값) 사이의 잔차 값을 저장하는 리스트변수 초기화

    mns_value = []
    mns_value.extend(x / 1000 for x in mns_coordi) # 알고리즘에는 [m]단위로 환산해서 들어감
    mns_value.extend(mns_b)                        # 자기밀도 값 집어넣음

    val = array_Val.reshape(3,3,3)           # 센서 값을 3x3 형태로 다시 저장(for 계산 용이)
    k_ij = []                                # K(i,j) 값을 저장할 리스트 변수 초기화
    hh = 0.118                               # 센서들 사이 떨어져있는 거리 h 초기화 (단위:m)
    k = 0                                    # k_ij 리스트의 index 변수 초기화

    # print(init_pos)

    # 각 센서마다 K_ij 값 계산 (논문의 식 11) -> 총 9개의 K_ij값이 계산됨
    # 센서 배열이 3x3이므로, 0보다 작거나 2보다 크면 해당 센서 위치의 값은 0으로 처리
    for i in range(3):
        for j in range(3):
            param = [0,0,0,0,0]
            param[0]=0 if j-1 < 0 else val[i][j-1][2]
            param[1]=0 if j+1 > 2 else val[i][j+1][2]
            param[2]=0 if i-1 < 0 else val[i-1][j][2]
            param[3]=0 if i+1 > 2 else val[i+1][j][2]
            param[4]=(-4)*(val[i][j][2])
            
            k_ij.append( (-1)*(sum(param) / (hh**2)) )  # K_ij 배열에 하나씩 추가

            buffer_residual = k_ij[k] - cal_BB(mns_value, P[k]) - cal_BB(init_pos, P[k])  # 실제값과 이론값 사이의 잔차 계산
            differences.append(buffer_residual)    # 각 센서들의 잔차 값을 differences 배열에 1차원으로 삽입
            k += 1


    # pprint.pprint(differences) # 계산한 잔차 값의 총합 출력
    return differences


# 자석의 자기밀도를 계산하는 함수 (논문의 식 15)
# A: 자석의 현재 위치좌표, P: 센서의 위치좌표, H: 자석의 자계강도
def cal_BB(A_and_H, P):
    global MU0
    A = [A_and_H[0], A_and_H[1], A_and_H[2]]   # 자석의 위치 값 따로 A 리스트에 저장
    M = [A_and_H[3], A_and_H[4], A_and_H[5]]   # 자석의 자기모멘트 벡터 값 따로 H 리스트에 저장
    R = np.array(P-A)                          # ij번째 센서와 자석 사이의 거리벡터
    Rn = np.linalg.norm(R)                     # ij번째 센서와 자석 사이의 거리(norm1)

    const = MU0 / (4*np.pi)     # 상수항 계산
    b1 = (9*M[2]) / (Rn ** 5)
    b2 = (45*(R[2])*(np.dot(M,R)+(M[2]*R[2]))) / (Rn ** 7)
    b3 = (105*(R[2]**3)*(np.dot(M,R))) / (Rn ** 9)

    return const*(b1-b2+b3)







#### Kalman Filter의 상태 천이 행렬을 반환하는 함수 ####
def fx(state, dt):
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
    
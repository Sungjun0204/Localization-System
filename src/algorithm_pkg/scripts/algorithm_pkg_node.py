#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import levenberg_marquardt as LM
import pprint
import sys
import signal


# 전역변수 선언
zero_setting_flag = 0           # 4~5번 뒤에 offset을 작동시키기 위한 flag변수 선언
#first_value = np.array([0.118, -0.118, 0.065]) # 위치 초기 값
first_value = [0.0, 0.0, 0.065, 1050000, 1050000, 0]
full_packet = ""                # 패킷 값을 저장하는 리스트 변수
sensor_data = []                # 해체작업을 진행할 패킷 값을 저장하는 리스트 변수
packet_count = 0                       # 분할되어 들어오는 패킷을 총 10번만 받게 하기 위해 카운트를 세는 변수
is_collecting = False
result = [0,0,0]                     # 최종적으로 추정한 위치 값을 저장하는 리스트 변수 선언

array_Val = np.array([  [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0] ])     # sensor 값들의 numpy조작을 간편하게 하기 위해 옮겨 저장할 배열 변수 선언
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
MU = 1.02                 # 사용하는 자석의 매질 투자율[-]
M0 = 1.320 / MU0          # 사용하는 자석의 등급에 따른 값[A/m]: 실험에서 사용하는 자석 등급은 N42 / 1.3[T]에서 [A/m]로 환산하기 위해 MU0 값을 나눔

#M_T = (np.pi)*(4.7625**2)*(12.7)*M0    # 자석의 자화벡터 값 = pi*(반지름^2)*(높이)*(자석 등급)
M_T = (np.pi)*(0.0047625**2)*(0.0127)*M0

flag = 0    # 알고리즘 첫 시작 때만 H벡터 정규화 진행을 하기 위한 플래그변수 선언
np.set_printoptions(precision=5, suppress=True)    # 배열 변수 출력 시 소수점 아래 5자리까지만 출력되도록 설정

OFFSET_TIME = 1




def signal_handler(signal, frame): # ctrl + c -> exit program
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)





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
                                                                # hall seneor는 단위가 uT이므로, mT로 단위 통일 (/1000)
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

    zero_setting_flag += 1                   # 이후 1로 설정하여 함수가 호출되지 않도록 초기화
                                            # 이렇게 안 하면 센서 측정 값이 계속 0으로만 출력됨


# offset 적용하는 함수
def offset_Setting():
    global array_Val, zero_Val, first_value, flag, result

    array_Val = np.array(array_Val) - np.array(zero_Val)    # offset 적용

    mean_vector = np.mean(array_Val, axis=0)    # 9개의 센서 값에 대한 평균
    norm_vector = np.linalg.norm(mean_vector)   # 계산한 평균 벡터의 norm을 계산

    if(norm_vector > 0.005):
        ### 본격적인 위치추정 코드 ###
        initial_guess = first_value    # 초기 자석의 위치좌표 및 자계강도 값
        
        # 먼저 초기값 H(내가 설정하는 값)를 정규화시켜야 함
        if(flag == 0):        # 알고리즘 첫 시작에만 정규화시키면 됨. 이 후로는 알아서 정규화 값을 기준으로 값을 추정할 것임
            initial_guess[5] = ((-1)*(initial_guess[3]*initial_guess[4]) / (initial_guess[3]+initial_guess[4]))  # Z값을 조건에 맞게 계산 후 대입 (m^2 + n^2 + p^2 = 1)
            flag = 1
        elif(flag == 1):
            initial_guess[5] = (1-(initial_guess[3]**2)-(initial_guess[4]**2))
            H_norm = np.linalg.norm(initial_guess[3:6])                    # 정규화 계산을 위한 norm값 계산
            initial_guess[3:6] = np.array(initial_guess[3:6]) / H_norm     # 초기 자계강도(H) 벡터에 대한 정규화 진행


        result_pos = least_squares(residuals, initial_guess, method='lm')    # Levenberg-Marquardt Algorithm 계산
        
        result = [result_pos.x[0]*100, result_pos.x[1]*100, result_pos.x[2]*100]  # 위치 근사값만 따로 저장
        # pprint.pprint(result * 100)                          # 위치 근사값 출력
        # print(result_pos)

        # 위치추정을 위한 초기값을 이전에 구한 추정값으로 초기화
        for i in range(6):
            first_value[i] = result_pos.x[i]



    else:
        print("!! Out of Workspace !!")


    # pretty_print(array_Val)   # 가공된 센서링된 값 출력 
    # mean_vector = np.mean(array_Val, axis=0)    # 9개의 센서 값에 대한 평균
    # print(np.linalg.norm(mean_vector))   # 계산한 평균 벡터의 norm을 출력



# 측정한 자기장 값과 계산한 자기장 값 사이의 차이를 계산하는 함수
# 여기서 오차 제곱까지 해 줄 필요는 없음. least_squares에서 알아서 계산해 줌
def residuals(init_pos):
    global array_Val, P
    differences = [] # 센서 값과 계산 값 사이의 잔차 값을 저장하는 배열변수 초기화

    # 위치에 대한 잔차 값의 총합 저장
    for i in range(9):
        buffer_residual = (array_Val[i] - np.array(cal_B(init_pos, P[i]))[:3])  # 실제값과 이론값 사이의 B값 잔차 계산
        differences.extend(buffer_residual)    # 각 센서들의 잔차를 순서대로 잔차배열에 삽입
        
        normalized_B = array_Val[i] / np.linalg.norm(array_Val[i])   # 측정한 자기장 값을 정규화
        differences.extend(normalized_B - np.array(cal_B(init_pos, P[i]))[3:6])     # 정규화된 측정 자기장 값과 예상 자기장 값 사이의 차이 계산 후 순서대로 잔차배열에 삽입

    # print(differences)
    return differences    # 최종적으로 6x9=54개의 잔차값이 저장된 리스트를 반환




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
    return [b_x, b_y, b_z, H[0], H[1], H[2]]            # 최종 자기밀도 값 반환 (자계강도 값은 앞서 계산한 값 그대로 저장)

# 두 점 사이의 거리를 구하는 함수
def distance_3d(point1, point2):
    # return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)**0.5
    return np.linalg.norm(point1-point2)

# H dot P의 값을 계산하는 함수(논문 식(3)~(5))
def h_dot_p(A, H, P):
    return (H[0]*(P[0]-A[0])) + (H[1]*(P[1]-A[1])) + (H[2]*(P[2]-A[2]))






# 메인 함수
def main():

    global array_Val, result


    rospy.init_node('algorithm_pkg_node', anonymous=True)   # 해당 노드 기본 설정
    
    #### 메세지 발행 설정 구간 ####
    # pub = rospy.Publisher('predicted_local', PointStamped, queue_size=10)   # 최종 추정한 자석의 위치좌표
    pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    #### 메세지 구독 설정 구간 ####
    rospy.Subscriber('read', String, seperating_Packet)   # /read를 구독하고 seperating_Packet 함수 호출: 패킷 처리 함수
    rospy.Subscriber('Is_offset', String, callback_offset)  # /Is_offset을 구독하고 callback_offset 함수 호출

    rate = rospy.Rate(10)  # 10Hz


    #### 메인 반복문 ####
    while (not rospy.is_shutdown()):
        marker = Marker()
        marker.header = Header(frame_id="map", stamp=rospy.Time.now())
        marker.ns = "my_namespace"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = result[0]
        marker.pose.position.y = result[1]
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
        pub.publish(marker)

        rate.sleep()

    rospy.spin()    # node 무한 반복


if __name__ == '__main__':
    main()                  # main문 무한 호출
    

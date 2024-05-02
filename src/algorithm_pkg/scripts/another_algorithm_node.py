#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from scipy.optimize import least_squares
import numpy as np
import pprint
import sys, signal


# 전역변수 선언
zero_setting_flag = 0           # 4~5번 뒤에 offset을 작동시키기 위한 flag변수 선언
first_value = [0.0, 0.0, 0.065, 0, 0, 8.61e-7] # 0~2번째는 위치 초기 값, 3~5번째는 자석의 자기모멘트 벡터(m) 초기 값
full_packet = ""                # 패킷 값을 저장하는 리스트 변수
sensor_data = []                # 해체작업을 진행할 패킷 값을 저장하는 리스트 변수
packet_count = 0                # 분할되어 들어오는 패킷을 총 10번만 받게 하기 위해 카운트를 세는 변수
is_collecting = False           #         
result = [0,0,0]                # 최종적으로 추정한 위치 값을 저장하는 리스트 변수 선언
flag = 0                        # 알고리즘 첫 시작 때만 H벡터 정규화 진행을 하기 위한 플래그변수 선언


array_Val = np.array([  [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0] ])    # sensor 값들의 numpy조작을 간편하게 하기 위해 옮겨 저장할 배열 변수 선언

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

np.set_printoptions(precision=5, suppress=True)    # 배열 변수 출력 시 소수점 아래 5자리까지만 출력되도록 설정





##################### 프로그램 강제종료를 위한 코드 ########################

def signal_handler(signal, frame): # ctrl + c -> exit program
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

######################################################################




# 함수 선언


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

######################################################################################################################






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
    global array_Val, zero_Val, first_value, flag, result
    result = [0, 0, 0]                          # 최종 추정 값들 중, 자석의 위치 값만 저장할 리스트 변수 초기화
    array_Val = np.array(array_Val) - np.array(zero_Val)    # offset 적용
    mean_vector = np.mean(array_Val, axis=0)    # 9개의 센서 값에 대한 평균
    norm_vector = np.linalg.norm(mean_vector)   # 계산한 평균 벡터의 norm을 계산


    # 평균 norm 값이 0.005 이상이면(=자석이 센서 배열에서부터 20cm 이내로 위치하면)
    if(norm_vector > 0.005):
        ### 본격적인 위치추정 코드 ###
        initial_guess = first_value    # 초기 자석의 위치좌표 및 자계강도 값

        # import pdb; pdb.set_trace()

        result_pos = least_squares(residuals, initial_guess, method='lm')    # Levenberg-Marquardt Algorithm 계산
        
        result = [result_pos.x[0]*100, result_pos.x[1]*100, result_pos.x[2]*100]  # 위치 근사값만 따로 저장(미터 단위이므로, 보기 쉽게 cm단위로 환산하여 저장)
        pprint.pprint(result)      # 위치 근사값 출력
        # print(result_pos)
        # print("... Measuring ...")
        
        # 위치추정을 위한 초기값을 방금 구한 추정값으로 초기화
        for i in range(6):
            first_value[i] = result_pos.x[i]


    # 평균 norm 값이 0.005 이하면(=자석이 센서 배열에서부터 20cm 이상 떨어져 있으면)
    else:
        print("!! Out of Workspace !!") # 해당 텍스트만 출력하고 알고리즘 작동은 일절 없음




# 측정한 자기장 값과 계산한 자기장 값 사이의 차이를 계산하는 함수
# (여기서 오차 제곱까지 해 줄 필요는 없음. least_squares에서 알아서 계산해 줌)
def residuals(init_pos):
    global array_Val, P, first_value
    differences = []                        # (센서 값)과 (계산 값) 사이의 잔차 값을 저장하는 리스트변수 초기화

    val = array_Val.reshape(3,3,3)           # 센서 값을 3x3 형태로 다시 저장(for 계산 용이)
    k_ij = []                                # K(i,j) 값을 저장할 리스트 변수 초기화
    hh = 0.118                               # 센서들 사이 떨어져있는 거리 h 초기화 (단위:m)
    k = 0                                    # k_ij 리스트의 index 변수 초기화

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

            buffer_residual = k_ij[k] - cal_BB(init_pos, P[k])  # 실제값과 이론값 사이의 잔차 계산
            differences.append(buffer_residual)    # 각 센서들의 잔차 값을 differences 배열에 1차원으로 삽입
            k += 1

    # # 위치에 대한 잔차 값의 총합 저장
    # for i in range(9):
    #     buffer_residual = k_ij[i] - cal_BB(init_pos, P[i])  # 실제값과 이론값 사이의 잔차 계산
    #     differences.append(buffer_residual)    # 각 센서들의 잔차 값을 differences 배열에 1차원으로 삽입

    # pprint.pprint(differences) # 계산한 잔차 값의 총합 출력
    return differences


# 자석의 자기밀도를 계산하는 함수 (논문의 식 15)
# A: 자석의 현재 위치좌표, P: 센서의 위치좌표, H: 자석의 자계강도
def cal_BB(A_and_H, P):
    global MU0
    A = [A_and_H[0], A_and_H[1], A_and_H[2]]   # 자석의 위치 값 따로 A 리스트에 저장
    M = [A_and_H[3], A_and_H[4], A_and_H[5]]   # 자석의 자기모멘트 벡터 값 따로 H 리스트에 저장
    R = np.array(A-P)                          # ij번째 센서와 자석 사이의 거리벡터
    Rn = np.linalg.norm(R)                     # ij번째 센서와 자석 사이의 거리(norm1)

    const = MU0 / (4*np.pi)     # 상수항 계산
    b1 = (9*M[2]) / (Rn ** 5)
    b2 = (45*(R[2])*(np.dot(M,R)+(M[2]*R[2]))) / (Rn ** 7)
    b3 = (105*(R[2]**3)*(np.dot(M,R))) / (Rn ** 9)

    return const*(b1-b2+b3)






# 메인 함수
def main():

    global array_Val, result


    rospy.init_node('algorithm_pkg_node', anonymous=True)   # 해당 노드의 기본 설정

    #### 메세지 발행 설정 구간 #### 
    pub = rospy.Publisher('visualization_marker', Marker, queue_size=10) # 최종 추정한 자석의 위치좌표

    #### 메세지 구독 설정 구간 ####
    rospy.Subscriber('read', String, seperating_Packet)   # /read를 구독하고 seperating_Packet 함수 호출: 패킷 처리 함수
    rospy.Subscriber('Is_offset', String, callback_offset)  # /Is_offset을 구독하고 callback_offset 함수 호출

    rate = rospy.Rate(1000)  # 10Hz

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
        marker.pose.position.z = result[2]
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
    

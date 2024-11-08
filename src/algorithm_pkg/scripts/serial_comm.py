#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import serial
import numpy as np
from std_msgs.msg import Float32MultiArray

actuation_matrix = np.zeros([3,2])
previous_float_value = []

MU0 = 4*(np.pi)*(1e-7)    # 진공투자율[H/m]

# 5번 센서 기준으로 측정한 C-Mag MNS의 추정 자기모멘트와 위치벡터 값
# c1_m = np.array([-8955.11799  , 287.65482 ,  6153.42911])
# c2_m = np.array([-12030.41578 , -25817.64688 , -10339.12136])
# c1_r = np.array([0.01524, 0.06372, 0.11369])
# c2_r = np.array([-0.00854,0.04312,-0.09347])
#
# c1_m = np.array([9.75471641 , 17.62471106, -24.43889954])
# c2_m = np.array([-2.0485679,  -25.27196832,  -8.81304558])
# c1_r = np.array([-0.02055655,  0.03727525,  0.2])
# c2_r = np.array([0.01518364, 0.10440898, 0.2])
#
# c1_m = np.array([1000.   ,     999.99999 , 1000.00604])
# c2_m = np.array([312.33744 , 36164.36643 , 23929.72647])
# c1_r = np.array([-0.01503  ,  -0.0666   ,   0.11295])
# c2_r = np.array([ 0.00468  ,   -0.04062  ,    0.19999])

# 각각 원점에서 측정한 경우
c1_m = np.array([ -1860.93027 ,   -50494.0572 ,  18030.43832])
c2_m = np.array([44763.18064  ,  545.57621 ,  -5192.68544])
c1_r = np.array([ 0.02219  ,    0.15341  ,    0.2])
c2_r = np.array([0.13445  ,   -0.0589   ,   0.2])

## C-Mag MNS의 추정 위치벡터와 자계강도 값
c1_r = np.array([0.00366, -0.00263,  0.15274])
c2_r = np.array([-0.00777, -0.42695 , 0.49411])
c1_h = np.array([ -0.00032,  0.00087, -0.05164])
c2_h = np.array([-0.01112, -0.58521,  0.47029]) 



# 5번 센서 기준으로 측정한 C-Mag MNS의 추정 자기모멘트 값22
# c1_b = np.array([1446.35, 2485.6, 2760.53])
# c2_b = np.array([-2664.57, -14622.23, -7922.74])

# c1_h = np.array([0.02075, -0.01221, 0.0083])
# c2_h = np.array([0.01482, 0.01928, 0.08779]) 



## 2번 센서가 C1 솔레노이드, 1번 센서가 C2 솔레노이드 전류 값을 읽고 있음
def process_serial_data(serial_data):
    global previous_float_value
    
    try:
        # 패킷 시작 확인
        if not serial_data.startswith('AAA'):
            return previous_float_value  # 오류 발생 시 이전 값 반환
        
        # 데이터 부분과 체크섬 추출
        parts = serial_data[3:].split(',')
        data_parts = parts[:5]
        checksum_part = int(parts[5])
        
        # 데이터 변환
        float_values = []
        calculated_checksum = 0
        for data_str in data_parts:
            # 부호 처리 및 정수 변환
            multiplier = -1 if data_str[0] == '1' else 1
            value = int(data_str[1:]) * multiplier
            float_values.append(value / 1000.0)
            calculated_checksum += abs(int(data_str[1:]))  # 체크섬 계산
        
        # 체크섬 검증
        if calculated_checksum != checksum_part:
            return previous_float_value  # 오류 발생 시 이전 값 반환
    
    except Exception as e:
        # print("Error occurred:", e)  # 오류 메시지 출력
        return previous_float_value  # 오류 발생 시 이전 값 반환
    
    previous_float_value = float_values
    return float_values





def actuation_callback(data):
    global actuation_matrix
    temp = np.array(data.data).reshape(3,2)
    # actuation_matrix = (temp)
    # actuation_matrix = np.array([[1.4377, 1.4377],[0.7557, 0.7557],[0, 0]])  # 원래 Actuation 행렬
    actuation_matrix = np.array([[1.836, 1.782],[0.306, 0.297],[0, 0]])
    # print(actuation_matrix)


# 메인문
def serial_comm():
    global actuation_matrix, previous_float_value

    # node 설정
    rospy.init_node('serial_comm', anonymous=True)
    
    # 발행 정보 #
    pub = rospy.Publisher('c_mag_b', Float32MultiArray, queue_size=10)
    
    # 구독 정보 #
    rospy.Subscriber('c_mag_actuation', Float32MultiArray, actuation_callback)

    rate = rospy.Rate(1000)  # 10Hz

    ser = serial.Serial(port="/dev/ttyACM0", baudrate=115200, timeout=1)
    # ser = serial.Serial(port="/dev/ttyUSB2", baudrate=115200, timeout=1)

    while not rospy.is_shutdown():
        line = ser.readline()  # 시리얼 포트로부터 데이터를 읽어옵니다.
        if line:
            # 데이터 처리
            processed_data = process_serial_data(line.decode('utf-8', errors='ignore'))  # 전류 센서 값 받아 옴
            processed_array = np.array(processed_data)  # 배열 형태로 초기화
            # print(processed_data, previous_float_value)

            # 원래 방법
            current = np.array([processed_array[1], processed_array[0]]) # 1, 2번째 전류 값만 따로 취해서 저장
            mns_final_b = np.dot(actuation_matrix, current) # actuation 행렬과 전류 행렬 내적하여 MNS가 발생시키는 자기밀도 값 계산
            
            ### C-Core를 두 개의 솔레노이드로 분리 후 계산한 방법
            # mns_c1 = np.append(c1_r, (c1_h * processed_array[1]))
            # mns_c2 = np.append(c2_r, (c2_h * processed_array[0]))
            # mns_final_b = np.append(mns_c1, mns_c2)
                            # 원래는 actuation matrix 만들어서 곱해야 하는데
                            # 그게 안 되니까 이렇게 전류 값 받아서 단순 근사

            ### 솔레노이드를 두 개의 솔레노이드로 분리하여 하나로 합치는 방법
            # mns_final_b = (c1_m * processed_array[1]) + (c2_m * processed_array[0])

            # ROS 토픽으로 발행
            msg = Float32MultiArray()  # 얘 1차원 배열만 받을 수 있음
            msg.data = mns_final_b
            pub.publish(msg)
            # rospy.loginfo("{}".format(processed_data))
            print(processed_data)
            
            # print("... c mag b calculating ...")


        rate.sleep()

    ser.close()

if __name__ == '__main__':
    try:
        serial_comm()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import serial
import numpy as np
from std_msgs.msg import Float32MultiArray

actuation_matrix = np.zeros([3,2])
previous_float_value = []

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
    actuation_matrix = temp
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

    rate = rospy.Rate(100)  # 10Hz

    ser = serial.Serial(port="/dev/ttyACM0", baudrate=115200, timeout=1)

    while not rospy.is_shutdown():
        line = ser.readline()  # 시리얼 포트로부터 데이터를 읽어옵니다.
        if line:
            # 데이터 처리
            processed_data = process_serial_data(line.decode('utf-8'))
            processed_array = np.array(processed_data)
            # print(processed_data, previous_float_value)

            current = np.array([processed_array[0], processed_array[1]]) # 1, 2번째 전류 값만 따로 취해서 저장
            mns_final_b = actuation_matrix.dot(current) # actuation 행렬과 전류 행렬 내적하여 MNS가 발생시키는 자기밀도 값 계산

            # ROS 토픽으로 발행
            msg = Float32MultiArray()
            msg.data = mns_final_b
            pub.publish(msg)
            # rospy.loginfo("Published: {}".format(processed_data))
            
            print("... c mag b calculating ...")


        rate.sleep()

    ser.close()

if __name__ == '__main__':
    try:
        serial_comm()
    except rospy.ROSInterruptException:
        pass

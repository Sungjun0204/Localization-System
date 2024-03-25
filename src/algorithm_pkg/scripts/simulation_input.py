#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32MultiArray
import sys
import signal


# 원활한 강제종료를 위한 코드
def signal_handler(signal, frame): # ctrl + c -> exit program
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# 메인 함수
def simulation_input():
    pub = rospy.Publisher('initial', Float32MultiArray, queue_size=10)
    rospy.init_node('simulation_input', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    while not rospy.is_shutdown():
        # Prompt the user for input
        input_str = input("Enter three floats separated by spaces: ")
        input_list = list(map(float, input_str.split()))
        
        # Ensure the input list has three floats
        if len(input_list) != 3:
            print("Please enter exactly three floats.")
            continue
        
        # Create a Float32MultiArray message
        array_msg = Float32MultiArray()
        array_msg.data = input_list
        
        # Publish the message
        # rospy.loginfo(array_msg)
        pub.publish(array_msg)
        
        rate.sleep()


if __name__ == '__main__':
    try:
        simulation_input()
    except rospy.ROSInterruptException:
        pass

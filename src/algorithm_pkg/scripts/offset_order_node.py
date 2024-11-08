#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
import sys
import signal
import numpy as np



def signal_handler(signal, frame): # ctrl + c -> exit program
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)



def offset_order_node():

    pub = rospy.Publisher('Is_offset', String, queue_size=10)
    rospy.init_node('offset_order_node', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():

        x = input("Do you want to set the offset, input 1: ")
        if(x == 1):
            pub.publish("1")
            rate.sleep()




        # x = raw_input("Input task number or type 'help': ")

        # if(x == "help"):
        #     print("\n[-- Help Guide --]\n")
        #     print("10: offset setting mode")
        #     print("11~19: 0~8th sensor's data sampling")
        #     print("20: print the captured sensor set data")
        #     print("21: real time offset application mode\n")

        # elif x.isdigit():  # 숫자인지 확인
        #     pub.publish(x)
        #     rate.sleep()

        # else:
        #     print("-- Press the right number or type 'help' --")

        

if __name__ == '__main__':
    try:
        offset_order_node()
    except rospy.ROSInterruptException:
        pass

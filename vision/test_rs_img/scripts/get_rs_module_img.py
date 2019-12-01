#!/usr/bin/env python3

import sys
sys.path.insert(1, "/home/iarc/.local/lib/python3.5/site-packages/")
import rospy
import cv2
from get_rs_image import Get_image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

if __name__ == '__main__':
    rospy.init_node('get_d435i_module_image', anonymous=True)
    listener = Get_image()
    
    while not rospy.is_shutdown():
        listener.display_mode = 'rgb'
        
        if(listener.display_mode == 'rgb')and(type(listener.cv_image) is np.ndarray):
            cv2.imshow("rgb module image", listener.cv_image)
        elif(listener.display_mode == 'depth')and(type(listener.cv_depth) is np.ndarray): 
            cv2.imshow("rgb module image", listener.cv_depth)
        else:
            pass
        cv2.waitKey(1)

    rospy.spin()


#!/usr/bin/env python3

import sys
sys.path.insert(1, "/home/iarc/.local/lib/python3.5/site-packages/")
import rospy
import cv2
from get_rs_image import Get_image
from mrcnn import Mask_RCNN
from cv_bridge import CvBridge, CvBridgeError
import numpy as np



if __name__ == '__main__':
    rospy.init_node('get_d435i_module_image', anonymous=True)
    sub_img = Get_image()
    mrcnn1 = Mask_RCNN()

    while not rospy.is_shutdown():
        print('========================================================')
        mrcnn1.predict_image([sub_img.cv_image])
        cv2.waitKey(1)
    rospy.spin()


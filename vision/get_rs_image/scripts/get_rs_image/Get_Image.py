#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
# sys.path.insert(1, "/usr/local/lib/python3.5/dist-packages/")

sys.path.insert(1, "/home/iarc/.local/lib/python3.5/site-packages/")
# sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import rospy
from sensor_msgs.msg import Image
from get_rs_image.srv import *
#sys.path.insert(1, "/home/iclab-arm/.local/lib/python3.5/site-packages/") 
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class Get_image():
    def __init__(self):
            
        self.bridge = CvBridge()
        self.image = np.zeros((0,0,3), np.uint8)
        self.depth = np.zeros((0,0,3), np.uint8)
        self.take_picture_counter = 0

        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        # rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        # rospy.Subscriber("/camera/aligned_depth_to_infra1/image_raw", Image, self.depth_callback)
        # rospy.Subscriber("/camera/depth_registered/points", Image, self.depth_callback) 

        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)

        self.cv_image = None
        self.cv_depth = None
        self.display_mode = 'rgb'
        # self.display_mode = 'depth'

    def show_image(self):
        '''
        Note: Call this function only when you directly execute this node (for test).
               Do not call this function when you use this py-class as an object. 
        '''
        image_dim = np.asarray(self.cv_image).shape
        depth_dim = np.asarray(self.cv_depth).shape
        print([image_dim, depth_dim])
        
        if(self.display_mode=='rgb'):
            cv2.imshow("rgb result", self.cv_image)

        elif(self.display_mode=='depth'):
            cv2.imshow("depth result", self.cv_depth)

        else:
            print('unknow mode')
            pass
        cv2.waitKey(1)

    def rgb_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")    # for rgb image

        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        try:
            if(self.display_mode == 'depth'):
                tmp = self.bridge.imgmsg_to_cv2(data, "16UC1")    # for depth image 16UC1
                self.cv_depth = cv2.applyColorMap(cv2.convertScaleAbs(tmp, alpha=0.03), cv2.COLORMAP_JET)
            else:
                self.cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")    # for depth image 16UC1

        except CvBridgeError as e:
            print(e)


    def Cal_depth_normal(self, depth_img):
        '''
        Usage : Calculate the normal vectors map of a depth image.
        Input : A depth image.
        Output: The normal vectors map of input depth image.
        '''
        use_grad = 0
        if(use_grad == 1):
            zy, zx = np.gradient(depth_img)  
            normal = np.dstack((-zx, -zy, np.ones_like(depth_img)))
        else:
            zx = cv2.Sobel(depth_img, cv2.CV_64F, 1, 0, ksize=3)     
            zy = cv2.Sobel(depth_img, cv2.CV_64F, 0, 1, ksize=3)
            normal = np.dstack((zx, zy, np.ones_like(depth_img)))
        
        # normalize to unit vector
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

        return normal

    def Get_depth_normal(self, normal_vec_map, coor_list):
        '''
        Usage : Get specified depth normal vectors. 
        Input : 2D coordinates of desire normal vectors.
        Output: Normal vectors of the input coordintes.
        '''
        Get_normal_vectors_from_depth = lambda coor2D: normal_vec_map[ coor2D[1], coor2D[0] ]
        tmp = map(Get_normal_vectors_from_depth, coor_list)
        depth_normals = list(map(list,  tmp))
        return depth_normals

# if __name__ == '__main__':
    # print('python version is: ', sys.version)
    # rospy.init_node('get_image_from_rs_d435i', anonymous=True)
    # listener = Get_image()
    # rospy.spin()
    # cv2.destroyAllWindows()
    

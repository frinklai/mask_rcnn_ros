#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import rospy
sys.path.insert(1, "/home/iarc/.local/lib/python3.5/site-packages/")
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from mrcnn.config import Config
from datetime import datetime
from get_rs_image import Get_image

# Root directory of the project
ROOT_DIR = os.getcwd() + '/mask_rcnn/'
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"shapes20191115T1117/mask_rcnn_shapes_0060.h5")    

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("MODEL_PATH not exists!!", COCO_MODEL_PATH)
 
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
 
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + obj_class_num
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
class Mask_RCNN():
    def __init__(self):
        self.obj_bboxes = []
        self.obj_scores = []
        self.obj_masks  = []
        self.class_name = []
        self.mask_image = []

        self.Enable_display = True
        self.config = InferenceConfig()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode = "inference", model_dir = MODEL_DIR, config = self.config)
        
        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        
        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'bottle', 'screw_driver']

        # Load a random image from the images folder
        # file_names = next(os.walk(IMAGE_DIR))[2]

    def convert_id2label(self, id):
        return self.class_names[id]

    def predict_image(self, image):

        start = time.time()
        # Run detection
        tmp_results = self.model.detect(image, verbose=1)
        end = time.time()

        fps = 1 / (end-start)
        # print('=============')
        # print('Mask R-CNN FPS = ', round(fps, 2))
        # print('=============')
        
        results = tmp_results[0]
        self.obj_bboxes = results['rois']
        self.obj_scores = results['scores']
        self.obj_masks  = results['masks']
        self.class_name = list(map(self.convert_id2label, results['class_ids']))
        self.class_name = np.asarray(self.class_name)

        # print('*** class = ',self.class_name , '***')
        # if(self.Enable_display==True):
        # self.Enable_display = False
        self.mask_image = visualize.display_instances(image, self.obj_bboxes, self.obj_masks, self.class_name, 
                                                      self.Enable_display, self.obj_scores)


if __name__ == '__main__':
    rospy.init_node('mask_rcnn_predict', anonymous=True)
    sub_img = Get_image()
    mrcnn = Mask_RCNN()
    img = np.asarray(sub_img.cv_image)

    #################################
    #  Start load image to predict  #
    #################################
    while not rospy.is_shutdown():
        print('========================================================')
        mrcnn.predict_image([sub_img.cv_image])
        cv2.waitKey(1)
    rospy.spin()



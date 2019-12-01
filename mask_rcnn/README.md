# ROS-based Mask R-CNN for Object Detection and Segmentation
This package is forked from https://github.com/matterport/Mask_RCNN

## **Running mask rcnn by ros (realsense d435i image as input)**

### Environment setting
- Remember to change python path in `train.py` and `predict.py`
- change this: `sys.path.insert(1, "/home/<your_pc_name>/.local/lib/python3.5/site-packages/")`
- python path may be different for different pc, please check your own path
- Make sure `__init__.py` is in `<ros_ws>/src/mask_rcnn_ros/mask_rcnn/scripts/mrcnn` 

### Running as a ros node
```  
cd src/mask_rcnn_ros/vision/
source create_catkin_ws.sh
cd to <ros_ws>
. /devel/setup.bash
roslaunch realsense2_camera rs_rgbd.launch
```

open new terminal    

```
. /devel/setup.bash 
. ../catkin_workspace/install/setup.bash --extend
cd <ros_ws>/src/mask_rcnn_ros
rosrun test_rs_img mrcnn_module.py 
```

## **Train custom mrcnn**

### 0. move __init__.py to a tmp folder
  - ex: from `<ros_ws>/src/mask_rcnn_ros/mask_rcnn/scripts/mrcnn`
   to `<ros_ws>/src/mask_rcnn_ros/mask_rcnn/scripts/mrcnn/tmp`

  - [NOTE] Do this step only if you suffered from an error when run train.py

### 1. cd 到`src/mask_rcnn_ros/mask_rcnn/train_data` 底下並建立如下的資料夾
  - cv2_mask
  - json
  - labelme_json
  - pic
  - viz_mask
  - orig_objs
  - NOTE: (train_data這個資料夾我有ignore掉, 所以需要自己建立)



### 2. 複製mrcnn_training_data_tools裡面的檔案到train_data下
  - 應該會有4個sh檔和1個py檔
  - 請注意工作路徑還是在`<ros_ws>/src/mask_rcnn_ros/mask_rcnn/train_data下`

### 3. 準備要標記的圖片
  - cd 到 <ros_ws>/src/mask_rcnn_ros/mask_rcnn/train_data下
  - 把之後要標記的照片貼到` <ros_ws>/src/mask_rcnn_ros/mask_rcnn/train_data/orig_objs/train_ds/ `下
    * (NOTE):該路徑下請 不要 放jpg之外的任何檔案
    * train_ds可換成其他你自己定義的名稱
  - run $` python3 rename_file.py <資料集名稱>`
    * ex: $ `python3 rename_file.py train_ds`
    * (這個程式會把train_ds內的檔案都重新命名成train_ds1.jpg, train_ds2.jpg...)
  

### 4. 用labelme標記圖片
  - cd 到 `<ros_ws>/src/mask_rcnn_ros/mask_rcnn/train_data`下
  - 開啟terminal並執行 $ labelme 並開始標記圖片
    * [labeme使用的簡易教學連結](https://www.youtube.com/watch?v=mUoYFzYrVTE&t=26s)
    * labelme產生的json檔放在`./train_data/orig_objs/train_ds`下


### 5. run $`. loop_labelme_json_to_dataset.sh <ds_name>`
  - 此範例中, ds_name即為train_ds
  - 此步驟會解碼json文件並產生訓練所需的檔案與圖片

### 6. run $`. transfer_mrcnn_data.sh <ds_name>`
  - 此步驟會把訓練所需要的圖片與檔案歸類到應該要處在的資料夾底下

### 7. `cd ../`

### 8. 一次建立所有物件所需的檔案與圖片
  - 可執行此步驟來取代步驟5和6
  - 至auto_transfer.sh中修改程式
  - 把原本的ds名稱改成你需要的即可
    * 此步驟其實就是寫個程式來自動執行步驟5和步驟6


### 9. Download "mask_rcnn_coco.h5" file
  - from [here](https://drive.google.com/drive/folders/1p-woBULxZGaVYf0-QjWxGqpx0E7CA7oZ)
  or [here](https://drive.google.com/drive/folders/1dtKrPgSak0kyErXBoJv0HRWh8k71jdf6)

  - Put under src/mask_rcnn_ros/mask_rcnn

### 10. 修改train.py裡面的程式碼
  - 修改類別數量(NUM_CLASSES), NUM_CLASSES = 1+類別數量, 1代表背景(約在line56))
  - 增加訓練的class名稱(在函數load_shapes裡面), 如
  ```
      * self.add_class("shapes", 1, "類別名稱1")
      * self.add_class("shapes", 2, "類別名稱2")
      * self.add_class("shapes", 3, "類別名稱3") ...
  ```
  - load類別資料, (在函數load_mask裡面)
  ```
      * for i in range(len(labels)):
          * if labels[i].find("類別名稱1") != -1:
              + labels_form.append("類別名稱1")

          * elif labels[i].find("類別名稱2") != -1:
              + labels_form.append("類別名稱2")

          * elif labels[i].find("類別名稱3") != -1:
              + labels_form.append("類別名稱3")
  ```

### 11. run $python3 train.py
```  
- cd <ros_ws>/src/mask_rcnn_ros
- python3 mask_rcnn/scripts/mrcnn/train.py
```

## Predict mrcnn in python
  ### 0. 修改predict.py
    - in <ros_ws>/src/mask_rcnn_ros/mask_rcnn/scripts/mrcnn

  ### 1. 指定要進行預測的權重的路徑
    - COCO_MODEL_PATH = ...
    - 訓練好的權重檔都放在<ros_ws>/src/mask_rcnn_ros/mask_rcnn/logs 裡面

  ### 2. 修改要預測的物件名稱
    - 修改類別數量(NUM_CLASSES), NUM_CLASSES = 1+類別數量, 1代表背景(約在line38))
    - self.class_names = ['BG', '物件1名稱', '物件2名稱', ...]
    - ex: self.class_names = ['BG', 'bottle', 'brush', 'can', 'carton', 'cup', 
                            'hammer', 'remote_control', 'stapler']

  ### 3. 執行predict.py
  ```  
  - cd <ros_ws>/src/mask_rcnn_ros
  - python3 mask_rcnn/scripts/mrcnn/predict.py 

  ```

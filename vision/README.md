# rs_d435i

### realsense driver installation
- offical url: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md

### realsense ros package installation
- offical url: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages

### Display image from d435i by using ros
```  
cd <ros_ws>/src/vision/
source create_catkin_ws.sh
cd ../..
. /devel/setup.bash
roslaunch realsense2_camera rs_rgbd.launch
```

open new terminal

```
. /devel/setup.bash 
. ../catkin_workspace/install/setup.bash --extend
rosrun get_rs_image Get_Image.py
```
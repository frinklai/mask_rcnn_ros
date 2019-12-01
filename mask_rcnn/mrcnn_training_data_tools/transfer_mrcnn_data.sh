#!/bin/bash

for i in {1..10};
do
    echo $i

    # =================================
    # copy img.png to ./pic
    # =================================
    img_ori_path=./orig_objs/$1/$1$i\_json/img.png
    img_tar_path=./pic
    img_ori_rename_path=$img_tar_path/img.png
    img_tar_rename_path=$img_tar_path/$1$i.png
    # echo $img_ori_path
    # echo $img_tar_path
    # echo $img_ori_rename_path
    # echo $img_tar_rename_path
    cp $img_ori_path $img_tar_path
    mv $img_ori_rename_path $img_tar_rename_path


    # =================================
    # copy label.png to ./pic
    # =================================
    label_ori_path=./orig_objs/$1/$1$i\_json/label.png
    label_tar_path=./cv2_mask
    label_ori_rename_path=$label_tar_path/label.png
    label_tar_rename_path=$label_tar_path/$1$i.png
    # echo $label_ori_path
    # echo $label_tar_path
    # echo $label_ori_rename_path
    # echo $label_tar_rename_path

    cp $label_ori_path $label_tar_path
    mv $label_ori_rename_path $label_tar_rename_path


    # =================================
    # copy label_viz.png to ./viz_mask
    # =================================
    label_viz_ori_path=./orig_objs/$1/$1$i\_json/label_viz.png
    label_viz_tar_path=./viz_mask
    label_viz_ori_rename_path=$label_viz_tar_path/label_viz.png
    label_viz_tar_rename_path=$label_viz_tar_path/$1$i.png
    # echo $label_ori_path
    # echo $label_tar_path
    # echo $label_ori_rename_path
    # echo $label_tar_rename_path

    cp $label_viz_ori_path $label_viz_tar_path
    mv $label_viz_ori_rename_path $label_viz_tar_rename_path



    # =================================
    # move json_file to ./json
    # =================================
    json_file_ori_path=./orig_objs/$1/$1$i.json
    json_file_tar_path=./json
    # echo $json_file_ori_path
    mv $json_file_ori_path $json_file_tar_path


    # =================================
    # move json_forder to labelme_jsond
    # =================================
    json_folder_orig_path=./orig_objs/$1/$1$i\_json
    json_folder_tar_path=./labelme_json
    # echo $json_folder_orig_path
    mv $json_folder_orig_path $json_folder_tar_path


done
    
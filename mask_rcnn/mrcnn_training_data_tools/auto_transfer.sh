#!/bin/bash

. loop_labelme_json_to_dataset.sh bottle
. transfer_mrcnn_data.sh bottle

. loop_labelme_json_to_dataset.sh brush
. transfer_mrcnn_data.sh brush

. loop_labelme_json_to_dataset.sh can
. transfer_mrcnn_data.sh can

. loop_labelme_json_to_dataset.sh carton
. transfer_mrcnn_data.sh carton

. loop_labelme_json_to_dataset.sh cup
. transfer_mrcnn_data.sh cup

. loop_labelme_json_to_dataset.sh hammer
. transfer_mrcnn_data.sh hammer

. loop_labelme_json_to_dataset.sh remote_control
. transfer_mrcnn_data.sh remote_control

. loop_labelme_json_to_dataset.sh stapler
. transfer_mrcnn_data.sh stapler

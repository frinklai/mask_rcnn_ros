#!/bin/bash
for((i=1;i<=10;i++))
do 
    s3=${i}
    labelme_json_to_dataset ./orig_objs/$1/$1$i.json
done
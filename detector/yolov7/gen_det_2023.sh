#!/bin/bash
secs=train

declare -A seq=(
    ['S04']="c001"
)

for key in "${!seq[@]}"; 
do 
    declare -a s=(${seq[${key}]})
    for cam in ${s[@]}
    do
        # echo ${secs} ${key} ${cam}
        python detect_mcmt.py --sec ${secs} --seq ${key} --cam ${cam} --weights ./weights/yolov7-e6e.pt --conf-thres 0.1 --agnostic --img-size 1280 --classes 0 --save-txt --save-conf --cfg_file $1
    done
done
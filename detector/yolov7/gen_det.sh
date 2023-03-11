#!/bin/bash
secs=train

declare -A seq=(
    ['S01']="c001 c002 c003 c004 c005"
    # ['S03']="c010 c011 c012 c013 c014 c015"
    # ['S04']="c016 c017 c018 c019 c020 c021 c022 c023 c024 c025 c026 c027 c028 c029 c030 c031 c032 c033 c034 c035 c036 c037 c038 c039 c040"
)

for key in "${!seq[@]}"; 
do 
    declare -a s=(${seq[${key}]})
    for cam in ${s[@]}
    do
        # echo ${secs} ${key} ${cam}
        python detect_mcmt.py --sec ${secs} --seq ${key} --cam ${cam} --weights ./weights/yolov7-e6e.pt --conf-thres 0.1 --agnostic --img-size 1280 --classes 2 5 7 --save-txt --save-conf --cfg_file $1
    done
done
#!/bin/bash
MCMT_CONFIG="mcmt_all.yml"

#### Run Detector ####

# Split frame from video
cd detector/
python gen_img_2023.py --config ${MCMT_CONFIG} --sec train

# Get patch of object
cd yolov7/
bash gen_det_2023.sh ${MCMT_CONFIG}

#### Extract ReID feature ####
cd ../../reid
python extract_image_feat.py --config "mcmt_reid1.yml" --sec train --seq S04
python extract_image_feat.py --config "mcmt_reid2.yml" --sec train --seq S04
python merge_reid_feat.py --sec train --seq S04

cd ../tracker/top1
gdown 1YdVCXOtlCFm97UzehGrWMfrKvm5UorSc
python run.py train S04
# python stat_occlusion.py scmt
# cp -r ./scmt ../../mcmt/data

# cd ../../mcmt
# bash run.sh
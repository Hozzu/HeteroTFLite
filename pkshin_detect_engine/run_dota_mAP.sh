#! /bin/bash

DATADIR=/home/root/datasets/dota_v1_val_458
MODELFILE=yolov11n-obb-int8.tflite


echo $MODELFILE

rm -f ./detection_result.json

taskset -c 4,5,6,7 ./pkshin_detect image model/$MODELFILE $DATADIR/labels.txt $DATADIR/images detection_result.json npu

if [ $? -eq 1 ];
then
        echo "Calculating mAP.."
        taskset -c 4,5,6,7 ./cal_mAP_dota.sh detection_result.json
fi

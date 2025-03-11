#! /bin/bash

DATADIR=/home/root/datasets/coco_val2017
MODELDIR=/home/root/model

MODELFILE=$MODELDIR/ssd_mobilenet_v2.tflitemxq


MODELFILES=($MODELDIR/yolov5nu.tflitemxq $MODELDIR/yolov5su.tflitemxq $MODELDIR/yolov5mu.tflitemxq $MODELDIR/yolov5lu.tflitemxq $MODELDIR/yolov5xu.tflitemxq $MODELDIR/yolov8n.tflitemxq $MODELDIR/yolov8s.tflitemxq $MODELDIR/yolov8m.tflitemxq $MODELDIR/yolov8l.tflitemxq $MODELDIR/yolov8x.tflitemxq $MODELDIR/yolov9c.tflitemxq)
PARAMS=()

#for MODELFILE in model/ssd_mobilenet_v1.tflitehef model/ssd_mobilenet_v2.tflitehef model/efficientdet-lite0.tflitehef model/efficientdet-lite1.tflitehef model/efficientdet-lite2.tflitehef  model/yolov11n.tflitehef model/yolov11s.tflitehef model/yolov11m.tflitehef model/yolov11l.tflitehef model/yolov11x.tflitehef
#for MODELFILE in model/yolov8n.tflitehef model/yolov8s.tflitehef model/yolov8m.tflitehef model/yolov8l.tflitehef model/yolov8x.tflitehef model/yolov9c.tflitehef
#for MODELFILE in $(find model -name "*.tflite" -exec echo {}"hef" \;)

#for index in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
#do


#for batch in 1 2 3 6 12 19 23 46 69
#do

#rm -f ./detection_result.json

#echo $MODELFILE
#taskset -c 4,5,6,7 ./pkshin_detect image $MODELFILE $DATADIR/labels.txt $DATADIR/images detection_result.json 0 25 0,1,1,1,1 -1,-1

#echo ${MODELFILES[$index]}
#taskset -c 4,5,6,7 ./pkshin_detect image ${MODELFILES[$index]} $DATADIR/labels.txt $DATADIR/images detection_result.json 0 25 ${PARAMS[index]}


#if [ $? -eq 1 ];
#then
#        echo "Calculating mAP.."
#        taskset -c 4,5,6,7 ./cal_mAP_coco.sh detection_result.json
#fi

#done
#done

for MODELFILE in
do

echo "GPU"
echo $MODELFILE
taskset -c 4,5,6,7 ./pkshin_detect image $MODELFILE $DATADIR/labels.txt $DATADIR/images detection_result.json 0 40 0,1,1,1,1 -1,-1

done

for MODELFILE in
do

echo "DSP"
echo $MODELFILE
taskset -c 4,5,6,7 ./pkshin_detect image $MODELFILE $DATADIR/labels.txt $DATADIR/images detection_result.json 0 40 1,0,1,1,1 -1,-1

done

for MODELFILE in $MODELDIR/yolov5nu.tflitemxq $MODELDIR/yolov5su.tflitemxq $MODELDIR/yolov5mu.tflitemxq $MODELDIR/yolov5lu.tflitemxq $MODELDIR/yolov5xu.tflitemxq $MODELDIR/yolov8n.tflitemxq $MODELDIR/yolov8s.tflitemxq $MODELDIR/yolov8m.tflitemxq $MODELDIR/yolov8l.tflitemxq $MODELDIR/yolov8x.tflitemxq $MODELDIR/yolov9c.tflitemxq
do

echo "A Mobilint NPU"
echo $MODELFILE

result=""
for i in {1..10}
do
result=$'$result\n$(taskset -c 4,5,6,7 ./pkshin_detect image $MODELFILE $DATADIR/labels.txt $DATADIR/images detection_result.json 0 40 1,1,0,1,1 0.001,0.001)'
done
awk 'BEGIN {tt=0 pt=0 tp=0 mt=0 ap=0} $1==pkshinresult {print $0 tt+=$2 pt+=$3 tp+=$4 mt+=$5 ap+=$6} END {print("Average Turnaround time:\t" tt/10 "\nAverage preprocess time:\t" pt/10 "\nAverage turnaround + postprocess time:\t" tp/10 "\nMaximum Turnaround time:\t" mt/10 "\nApplication latency:\t" ap/10 )}' $result

done


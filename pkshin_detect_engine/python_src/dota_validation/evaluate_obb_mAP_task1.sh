#!/bin/bash
rm -r predictions_txt/*
rm -r predictions_merged_txt/*

############################################################################################################
# Convert coco like json result to dota txt result format
############################################################################################################

result_path=/home/root/ghpark/result/detection_result_yolov8n-obb_float32.json

echo ===============
echo -e "[INFO] Convert json result to dota txt result format"
echo "[INFO] path: $result_path"
python3 convert_json2dota_eval.py \
    $result_path
echo -e "\nDone\n"

############################################################################################################
# evaluate mAP with dota txt result format
############################################################################################################

dataset_label_path=/home/root/datasets/dota_v1_val_458/labelTxt
dataset_list_path=/home/root/datasets/dota_v1_val_458/image_list.txt

echo ===============
echo -e "[INFO] Evalute yolov8n-obb result from tflite on ADCM\n"
python3 dotadevkit_eval/evaluate_obb_task1.py \
    ./predictions_txt \
    $dataset_label_path \
    $dataset_list_path
echo -e "\nDone\n"
echo ===============

############################################################################################################

# echo -e "[INFO] Evalute yolov8n-obb result from ultralytics framework\n"
# # evaluate mAP with dota txt result format
# python3 dotadevkit_eval/evaluate_obb_task1.py \
#     /home/ghpark/tflite_workspace/runs/obb/val_yolov8n-obb_DOTAv1_1024/predictions_txt \
#     $dataset_label_path \
#     $dataset_list_path
# echo -e "\nDone\n"

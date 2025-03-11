# Dota dataset evaluation kit

Toolkit for calculating mAP from inference results based on Dota dataset

Edited by Ghpark <ghpark@redwood.snu.ac.kr>  
Reference: dotadevkit - https://github.com/ashnair1/dotadevkit/tree/master

## 1. Build

Build dotadekit_eval  
[!Note] The poly_iou package requires compilation using Cython.

```bash
./build_dotadevkit_eval.sh
```

## 2.Convert and evaluate

2.1. Convert coco-like result data (json-format) to txt format of Dota dataset

```bash
python3 convert_json2dota_eval.py /path/to/result/[filename].json

# python3 convert_json2dota_eval.py \
# ./result/detection_result_yolov8n-obb_float32.json
```

2.2. Evaluate mAP

```bash
python3 dotadevkit_eval/evaluate_obb_task1.py \
    /path/to/prediction_txt_folder \
    /path/to/label_folder \
    /path/to/[image_list].txt

# Example python3 dotadevkit_eval/evaluate_obb_task1.py \
#     ./predictions_txt \
#     datasets/dota_v1_val_458/labelTxt \
#     datasets/dota_v1_val_458/iamge_list.txt

```

## 3. Bash script for 2.1 and 2.2

Filename: convert_and_evaluate_obb_mAP_task1.sh

Fix the part that says '## fix here'

```bash
#!/bin/bash
## convert_and_evaluate_obb_mAP_task1.sh

rm -r predictions_txt/*
rm -r predictions_merged_txt/*

############################################################################################################
# Convert coco like json result to dota txt result format
############################################################################################################

result_path=/path/to/result/[filename].json ## fix here
echo ===============
echo -e "[INFO] Convert json result to dota txt result format"
echo "[INFO] path: $result_path"
python3 convert_json2dota_eval.py \
    $result_path
echo -e "\nDone\n"

############################################################################################################
# evaluate mAP with dota txt result format
############################################################################################################

dataset_label_path=/path/to/label_folder ## fix here
dataset_list_path=/path/to/[data_filename_list].txt ## fix here
echo ===============
echo -e "[INFO] Evalute yolov8n-obb result from tflite on ADCM\n"
python3 dotadevkit_eval/evaluate_obb_task1.py \
    ./predictions_txt \
    $dataset_label_path \
    $dataset_list_path
echo -e "\nDone\n"
echo ===============
```

Then execute the shell scripts

```bash
./evaluate_obb_mAP_task1

```

Here's the result

```plaintext

===============
[INFO] Convert json result to dota txt result format
[INFO] path: /home/ghpark/tflite_workspace/ADCM/ghpark/result/detection_result_yolov8n-obb_float32.json
Saving predictions with DOTA format to /home/ghpark/tflite_workspace/ADCM/ghpark/dota_validation/predictions_txt...

Done

===============
[INFO] Evalute yolov8n-obb result from tflite on ADCM

mAP: 0.28853644240317955
classaps:  [53.3475628   6.09672636 44.20670324 66.96163027 36.26328652 26.27705628
 22.6665423  16.25393067  6.06060606  9.05045602 13.65398168 32.03096799
 24.73607038 41.65495223 33.54419081]

Done

===============
```

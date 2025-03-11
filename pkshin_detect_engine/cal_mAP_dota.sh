#! /bin/bash

#argument1: detection_result.json

rm -rf ./predictions_txt

python3 python_src/dota_validation/convert_json2dota_eval.py $1
python3 python_src/dota_validation/dotadevkit_eval/evaluate_obb_task1.py ./predictions_txt /home/root/datasets/dota_v1_val_458/labelTxt /home/root/datasets/dota_v1_val_458/image_list.txt


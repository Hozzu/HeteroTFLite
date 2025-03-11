#! /bin/bash

#argument1: detection_result.json

python3 python_src/coco_validation/main.py /home/root/datasets/coco_val2017/groundtruth.json $1


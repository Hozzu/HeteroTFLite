#!/bin/bash
# build poly_iou module of dotadevkit_eval 
cd dotadevkit_eval
python3 setup.py build_ext --inplace
rm -r build
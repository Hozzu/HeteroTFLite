from evaluate import task1
import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Invalid arguments. Please run 'python evaluate_obb_task1.py -h' for help."
        )
        sys.exit(1)

    if sys.argv[1] == "-h":
        print(
            "Usage: python evaluate_obb_task1.py \
            <detection folder path> <annotation path> <image info file path>"
        )
        print(
            "Example: python evaluate_obb_task1.py \
            /path/to/predictions_txt \
            /path/to/labelTxt_val_458 \
            /path/to/images_val_458.txt"
        )
        sys.exit(0)

    dota_detections_txt = sys.argv[1] + r"/Task1_{:s}.txt"
    label_annotations = sys.argv[2] + r"/{:s}.txt"
    images = sys.argv[3]
    task1.evaluate(dota_detections_txt, label_annotations, images, "1.0")


# /home/ghpark/tflite_workspace/ADCM/ghpark/dota_validation/predictions_txt
# /home/ghpark/tflite_workspace/util/validation/labelTxt_val_458
# /home/ghpark/tflite_workspace/util/validation/images_val_458.txt

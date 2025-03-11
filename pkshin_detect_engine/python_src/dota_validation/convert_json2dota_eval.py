import json
from pathlib import Path
import sys
import numpy as np

np.set_printoptions(precision=3, suppress=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Invalid arguments. \
            Please run 'python convert_json2dota_eval.py -h' for help."
        )
        sys.exit(1)

    if sys.argv[1] == "-h":
        print("Usage: python convert_json2dota_eval.py <prediction_json>")
        sys.exit(0)

    # get prediction json file
    pred_json = sys.argv[1]

    # get current directory
    save_dir = Path.cwd()

    data = json.load(open(pred_json))

    names = []
    for d in data["categories"]:
        names.append(d["name"])

    images_dict = {}
    for i in data["images"]:
        images_dict[i["id"]] = i["file_name"]

    pred_txt = save_dir / "predictions_txt"
    pred_txt.mkdir(parents=True, exist_ok=True)

    # if pred_txt.exists remove it
    if pred_txt.exists():
        for file in pred_txt.glob("*"):
            file.unlink()
            
    # create dummy txt files
    for name in names:
        name = name.replace(" ", "-")
        with open(f'{pred_txt / f"Task1_{name}.txt"}', "w") as f:
            f.write("")

    # Save split results
    #print(f"[Convert_json2dota_eval.py]: Saving predictions with DOTA format to {pred_txt}...")
    for d in data["annotations"]:
        image_id = images_dict[d["image_id"]].split(".")[0]
        score = float(d["score"])
        classname = names[d["category_id"] - 1].replace(" ", "-")
        p = d["poly"]
        with open(f'{pred_txt / f"Task1_{classname}"}.txt', "a") as f:
            f.writelines(
                f"{image_id} {score:.5f} {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f} {p[4]:.3f} {p[5]:.3f} {p[6]:.3f} {p[7]:.3f}\n"
            )
    
    #print(f"[Convert_json2dota_eval.py]: Saved at {save_dir}/predictions_txt/")
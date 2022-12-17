"""preprocess"""
import argparse
import json
import os

parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--dataset_name', type=str, choices=["imagenet2012"], default="imagenet2012")
parser.add_argument('--data_path', type=str, default='', help='eval data dir')
def create_label(result_path, dir_path):
    """
    create_label
    """
    dirs = os.listdir(dir_path)
    file_list = []
    for file in dirs:
        file_list.append(file)
    file_list = sorted(file_list)
    total = 0
    img_label = {}
    for i, file_dir in enumerate(file_list):
        files = os.listdir(os.path.join(dir_path, file_dir))
        for f in files:
            img_label[f] = i
        total += len(files)
    json_file = os.path.join(result_path, "imagenet_label.json")
    with open(json_file, "w+") as label:
        json.dump(img_label, label)
    print("[INFO] Completed! Total {} data.".format(total))

args = parser.parse_args()
if __name__ == "__main__":
    create_label('./preprocess_Result/', args.data_path)

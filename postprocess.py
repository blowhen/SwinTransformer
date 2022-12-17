"""postprocess for 310 inference"""
import argparse
import json
import os

import numpy as np
from mindspore.nn import Top1CategoricalAccuracy, Top5CategoricalAccuracy

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="result files path.")
parser.add_argument('--dataset_name', type=str, choices=["imagenet2012"], default="imagenet2012")
args = parser.parse_args()

def calcul_acc(lab, preds):
    return sum(1 for x, y in zip(lab, preds) if x == y) / len(lab)


if __name__ == '__main__':
    batch_size = 1
    top1_acc = Top1CategoricalAccuracy()
    rst_path = args.result_dir
    label_list = []
    pred_list = []
    file_list = os.listdir(rst_path)
    top5_acc = Top5CategoricalAccuracy()
    with open('./preprocess_Result/imagenet_label.json', "r") as label:
        labels = json.load(label)
    for f in file_list:
        label = f.split("_0.bin")[0] + ".JPEG"
        label_list.append(labels[label])
        pred = np.fromfile(os.path.join(rst_path, f), np.float32)
        pred = pred.reshape(batch_size, int(pred.shape[0] / batch_size))
        top1_acc.update(pred, [labels[label],])
        top5_acc.update(pred, [labels[label],])
    print("Top1 acc: ", top1_acc.eval())
    print("Top5 acc: ", top5_acc.eval())

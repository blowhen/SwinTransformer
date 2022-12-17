#!/bin/bash

if [ $# -lt 4 ]; then
    echo "Usage: bash ./scripts/run_infer_onnx.sh [ONNX_PATH] [DATASET_PATH] [DEVICE_TARGET] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
onnx_path=$(get_real_path $1)
dataset_path=$(get_real_path $2)
device_target=$3
device_id=$4

echo "onnx_path: "$onnx_path
echo "dataset_path: "$dataset_path
echo "device_target: "$device_target
echo "device_id: "$device_id

function infer()
{
    python ./eval_onnx.py --pretrained=$onnx_path \
                          --data_url=$dataset_path \
                          --device_target=$device_target \
                          --device_id=$device_id \
                          --swin_config="./src/configs/swin_tiny_patch4_window7_224.yaml" \
                          --batch-size=1 &> infer_onnx.log
}
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
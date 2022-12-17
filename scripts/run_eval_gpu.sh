#!/bin/bash

if [ $# -lt 3 ]
then
    echo "Usage: bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]"
exit 1
fi

export DEVICE_ID=$1
CONFIG_PATH=$2
CHECKPOINT_PATH=$3
export RANK_SIZE=1
export DEVICE_NUM=1

rm -rf evaluation_gpu
mkdir ./evaluation_gpu
cd ./evaluation_gpu || exit
echo  "start training for device id $DEVICE_ID"
env > env.log
python ../eval.py --device_target="GPU" --swin_config=$CONFIG_PATH --pretrained=$CHECKPOINT_PATH > eval.log 2>&1 &
cd ../


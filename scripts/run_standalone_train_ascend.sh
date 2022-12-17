#!/bin/bash

if [ $# -lt 2 ]
then
    echo "Usage: bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]"
exit 1
fi

export RANK_SIZE=1
export DEVICE_NUM=1
export DEVICE_ID=$1
CONFIG_PATH=$2

rm -rf train_standalone
mkdir ./train_standalone
cd ./train_standalone || exit
echo  "start training for device id $DEVICE_ID"
env > env.log
python -u ../train.py \
    --device_id=$DEVICE_ID \
    --device_target="Ascend" \
    --swin_config=$CONFIG_PATH > log.txt 2>&1 &
cd ../

#!/bin/bash

if [ $# -lt 3 ]
then
    echo "Usage: bash ./scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]"
exit 1
fi
BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export CONFIG_PATH=$1
export CUDA_VISIBLE_DEVICES="$3"
export RANK_SIZE=$2
export DEVICE_NUM=$2

rm -rf train_gpu
mkdir ./train_gpu
cd ./train_gpu || exit
env > env.log

mpirun --allow-run-as-root -n $2 \
    python ${BASEPATH}/../train.py  --device_target="GPU" \
    --swin_config=$CONFIG_PATH \
    --seed=47 > log.txt 2>&1 &
cd ../



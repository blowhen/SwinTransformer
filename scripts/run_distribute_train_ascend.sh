#!/bin/bash

if [ $# -lt 2 ]
then
    echo "Usage: bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]"
exit 1
fi
export RANK_TABLE_FILE=$1
CONFIG_PATH=$2
export RANK_SIZE=8
export DEVICE_NUM=8


cores=`cat /proc/cpuinfo|grep "processor" |wc -l`
echo "the number of logical core" $cores
avg_core_per_rank=`expr $cores \/ $RANK_SIZE`
core_gap=`expr $avg_core_per_rank \- 1`
echo "avg_core_per_rank" $avg_core_per_rank
echo "core_gap" $core_gap
for((i=0;i<RANK_SIZE;i++))
do
    start=`expr $i \* $avg_core_per_rank`
    export DEVICE_ID=$i
    export RANK_ID=$i
    export DEPLOY_MODE=0
    export GE_USE_STATIC_MEMORY=1
    end=`expr $start \+ $core_gap`
    cmdopt=$start"-"$end

    rm -rf train_parallel$i
    mkdir ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cp  *.py ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $i, device $DEVICE_ID rank_id $RANK_ID"
    env > env.log
    taskset -c $cmdopt python -u ../train.py \
    --device_target Ascend \
    --device_id $i \
    --swin_config=$CONFIG_PATH > log.txt 2>&1 &
    cd ../
done

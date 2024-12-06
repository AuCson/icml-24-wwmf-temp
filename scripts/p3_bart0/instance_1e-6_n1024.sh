#!/bin/bash

TASK=${1}
GPU_TYPE=${2}

for CONFIG in er
do
session_name=OCL_${TASK}_${CONFIG}_n1024
session_name="${session_name/\./-}"
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gres=gpu:2080:1 --nodelist ink-ellie --time 30:00:00 python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/${CONFIG}.yaml configs/p3/instance-bart0-base-ocl/replay_n1024.yaml --templates postfix=_lr1e-6_freq1_n1024/${TASK} --ocl_task ${TASK} || bash & bash"
echo "Created tmux session: ${session_name}"
done
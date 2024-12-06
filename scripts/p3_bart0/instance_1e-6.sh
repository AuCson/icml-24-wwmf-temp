#!/bin/bash

TASK=${1}
GPU_TYPE=${2}

for CONFIG in er vanilla mir
do
session_name=OCL_${TASK}_${CONFIG}
session_name="${session_name/\./-}"
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gres=gpu:${GPU_TYPE}:1 --time 12:00:00 python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/${CONFIG}.yaml --templates postfix=_lr1e-6/${TASK} --ocl_task ${TASK} || bash & bash"
echo "Created tmux session: ${session_name}"
done
#!/bin/bash

TASK=${1}
GPU_TYPE=${2}

for CONFIG in er vanilla mir100 mir500
do
session_name=OCL_${TASK}_${CONFIG}
session_name="${session_name/\./-}"
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gres=gpu:${GPU_TYPE}:1 --time 3:00:00 python ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/bart0-base-ocl/${CONFIG}.yaml --do_train --templates postfix=_lr1e-6/${TASK} --ocl_tasks ${TASK} || bash & bash"
echo "Created tmux session: ${session_name}"
done
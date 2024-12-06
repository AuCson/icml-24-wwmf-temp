#!/bin/bash


CONFIG=vanilla
GPU_TYPE=2080


for TASK in super_glue-cb super_glue-copa super_glue-rte
do
session_name=OCL_${TASK}_${CONFIG}_1e-4_large
session_name="${session_name/\./-}"
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gres=gpu:${GPU_TYPE}:1 --time 1-12 python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/${CONFIG}_bg100_large.yaml configs/p3/instance-bart0-base-ocl/optim_head_only.yaml configs/p3/instance-bart0-base-ocl/lr1e-4.yaml configs/p3/instance-bart0-base-ocl/steps.yaml --templates postfix=_lr1e-4_headonly_step30/${TASK} --ocl_task ${TASK} || bash & bash"

echo "Created tmux session: ${session_name}"
done
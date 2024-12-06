#!/bin/bash


CONFIG=vanilla
GPU_TYPE=2080

echo "greedy decoding and eval model, 30step, fixbos"

for TASK in super_glue-wsc.fixed winogrande-winogrande_xl anli hellaswag super_glue-wic
do
session_name=OCL_${TASK}_${CONFIG}_large
session_name="${session_name/\./-}"
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gres=gpu:${GPU_TYPE}:1 --time 1-12 python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/${CONFIG}_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml --templates postfix=_lr1e-6_step30_greedy_eval_fixbos/${TASK} --ocl_task ${TASK} || bash & bash"

echo "Created tmux session: ${session_name}"
done
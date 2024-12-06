#!/bin/bash



for TASK in anli super_glue-wic super_glue-wsc.fixed winogrande-winogrande_xl hellaswag
do
for CONFIG in vanilla_bg100
do
session_name=OCL_${TASK}_${CONFIG}
session_name="${session_name/\./-}"
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gres=gpu:2080:1 --nodelist ink-ellie --time 1-0 python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/${CONFIG}.yaml --templates postfix=_lr1e-6/${TASK} --ocl_task ${TASK} || bash & bash"
echo "Created tmux session: ${session_name}"
done
done
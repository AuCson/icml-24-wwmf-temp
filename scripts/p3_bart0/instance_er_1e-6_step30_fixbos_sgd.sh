#!/bin/bash


CONFIG=vanilla
GPU_TYPE=a6000

echo "greedy decoding and eval model, 30step, sgd"

for TASK in super_glue-cb super_glue-copa super_glue-rte
do
  for LR in 1e-2
  do

  session_name=OCL_${TASK}_${CONFIG}_SGD${LR}_large-er
  session_name="${session_name/\./-}"
  tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gres=gpu:${GPU_TYPE}:1 --time 2-12 python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-2.yaml configs/p3/instance-bart0-base-ocl/er.yaml  --templates postfix=_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK} || bash & bash"


  echo "Created tmux session: ${session_name}"
done
done
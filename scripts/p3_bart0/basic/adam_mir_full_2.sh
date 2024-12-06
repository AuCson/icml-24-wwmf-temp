#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, adam"

#TASKS=${1}


for TASK in  super_glue-rte super_glue-wsc.fixed

do
  for LR in 1e-6
  do

  # mir1k-distill
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/mir1k_distill.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_adam_full_delay_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}



  echo "Created tmux session: ${session_name}"
done
done
#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, sgd"

TASKS=${1}

for TASK in ${TASKS}
do
  echo ${TASK}

  LR=1e-6
  # vanilla
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n.yaml configs/p3/instance-bart0-base-ocl/replay_freq.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_lr${LR}_step30_adam_abl_greedy_eval_fixbos/${TASK} task=${TASK} --ocl_task ${TASK}


  LR=1e-2
  # vanilla
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n.yaml configs/p3/instance-bart0-base-ocl/replay_freq.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_lr${LR}_step30_sgd_abl_greedy_eval_fixbos/${TASK} task=${TASK} --ocl_task ${TASK}

done
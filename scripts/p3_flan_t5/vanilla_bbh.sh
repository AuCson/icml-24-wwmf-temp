#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, adam"

#TASKS=${1}

SHARD_IDX=0
SHARD_TOTAL=8


for TASK in bbh

do
  for LR in 1e-6
  do
  # vanilla
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/bbh_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
  --templates postfix=_vanilla_bbh-lr${LR}/${TASK}/ task=${TASK} --ocl_task ${TASK} --max_step 1000


done
  for LR in 1e-4
  do
  # vanilla
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/bbh_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
  --templates postfix=_vailla_bbh_lora-lr${LR}/${TASK} task=${TASK}  --ocl_task ${TASK} --max_step 1000




  done
  echo "Created tmux session: ${session_name}"

done

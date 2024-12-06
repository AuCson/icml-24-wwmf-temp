#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, adam"

#TASKS=${1}


for TASK in mmlu

do
  for LR in 1e-6
  do
  # vanilla
  python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml  configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml \
  configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
  configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
  --templates postfix=_step30_adam_full_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK} --update

python scripts/fpd_pred_split.py /home/xsjin/cl-analysis/runs/instance-p3-bart0-large/vanilla_bg100_step30_adam_full_greedy_eval_fixbos-lr1e-6/mmlu

  echo "Created tmux session: ${session_name}"
done
done
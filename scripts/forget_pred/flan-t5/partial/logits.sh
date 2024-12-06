#!/bin/bash

PART=head
LR=lr1e-3
STEP=30

SHARD_IDX=0
SHARD_TOTAL=8



for TASK in mmlu
do

  PART="head"
  STEP=30
  python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml \
  configs/mmlu/shards_text.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/instance-bart0-base-ocl/optims/${PART}.yaml \
  --templates postfix=_step30_l512p_adam_partial_greedy_eval_mmlu_text-${LR}/${PART}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --update




done


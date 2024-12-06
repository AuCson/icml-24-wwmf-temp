#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, adam"

#TASKS=${1}

SHARD_IDX=0
SHARD_TOTAL=8


for SHARD_IDX in 0
do
for TASK in mmlu

do
  for LR in 1e-3
  do
  # vanilla
  PART="head"
  STEP=100
  python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
  configs/mmlu/shards_text.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps${STEP}.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/instance-bart0-base-ocl/optims/${PART}.yaml \
  --templates postfix=_step${STEP}_l512_adam_partial_greedy_eval_mmlu_text-lr${LR}/${PART}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --update \
  --postfix "_8000"




#  PART="l22"
#  python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#  configs/mmlu/shards_text.yaml \
#  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
#  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/instance-bart0-base-ocl/optims/${PART}.yaml \
#  --templates postfix=_step30_l512_adam_partial_greedy_eval_mmlu_text-lr${LR}/${PART}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --update
#
#  PART="decoder"
#  python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#  configs/mmlu/shards_text.yaml \
#  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
#  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/instance-bart0-base-ocl/optims/${PART}.yaml \
#  --templates postfix=_step30_l512_adam_partial_greedy_eval_mmlu_text-lr${LR}/${PART}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --update
#
#  PART="head"
#  python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#  configs/mmlu/shards_text.yaml \
#  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
#  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/instance-bart0-base-ocl/optims/${PART}.yaml \
#  --templates postfix=_step30_l512_adam_partial_greedy_eval_mmlu_text-lr${LR}/${PART}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --update
#
#  PART="l18"
#  python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#  configs/mmlu/shards_text.yaml \
#  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
#  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/instance-bart0-base-ocl/optims/${PART}.yaml \
#  --templates postfix=_step30_l512_adam_partial_greedy_eval_mmlu_text-lr${LR}/${PART}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --update
#

  echo "Created tmux session: ${session_name}"
done
done
done
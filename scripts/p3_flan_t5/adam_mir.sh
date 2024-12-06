#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

SHARD_IDX=0
SHARD_TOTAL=8

for SHARD_IDX in 0 1 2
do
for TASK in mmlu

do
  for LR in 1e-6
  do
  # vanilla
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
  configs/mmlu/shards_text.yaml configs/p3/instance-bart0-base-ocl/mir1k_distill_flan_t5.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
  --templates postfix=_step30_l512_adam_full_greedy_eval_mmlu_text-lr${LR}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --max_step 1000



  echo "Created tmux session: ${session_name}"
done
done
done


#echo "greedy decoding and eval model, 30step, adam"
#
##TASKS=${1}
#
#
#for TASK in ${1}
#
#do
#  for LR in 1e-6
#  do
#  # vanilla
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/mir1k_distill_flan_t5.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_step30_adam_distill_er_full_greedy_eval-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK} --max_step 100
#
#
#
#  echo "Created tmux session: ${session_name}"
#done
#done
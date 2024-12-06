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


#  LR=1e-6
#
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#  configs/mmlu/shards_text_dev.yaml configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/mir1k_distill_flan_t5_cand64.yaml \
#  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
#  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#  --templates postfix=_continual_evtest_step30_f8n10_adam_full_greedy_eval_mmlu_text-lr${LR}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --max_step 1000 \
#  --do_test_eval

  LR=1e-4

#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#  configs/mmlu/shards_text_dev.yaml configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/mir1k_distill_flan_t5_cand64.yaml \
#  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n4.yaml \
#  configs/p3/instance-bart0-base-ocl/replay_freq5.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
#  --templates postfix=_step30_l512p_adam_continual_lora_f4n5_greedy_eval_mmlu_text-lr${LR}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --max_step 1000 \
#  --do_test_eval

  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml \
  configs/mmlu/shards_text_xl_dev.yaml configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/mir1k_distill_flan_t5_cand64.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n2.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq5.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
  --templates postfix=_step30_l512p_adam_continual_lora_f2n5_greedy_eval_mmlu_text-lr${LR}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --max_step 1000 \
  --do_test_eval --gc




done
done
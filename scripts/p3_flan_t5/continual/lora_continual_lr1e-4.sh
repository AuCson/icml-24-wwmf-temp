#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, adam"

#TASKS=${1}

SHARD_IDX=0
SHARD_TOTAL=8



for TASK in mmlu

do
  for LR in 1e-4
  do


  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
  configs/mmlu/shards_text_dev.yaml configs/p3/instance-bart0-base-ocl/continual.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
  --templates postfix=_step30_l512p_adam_lora_continual_greedy_eval_mmlu_text-lr${LR}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --max_step 1000 --gc



  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
  configs/mmlu/shards_text_dev.yaml configs/p3/instance-bart0-base-ocl/distill_er_flan_t5.yaml configs/p3/instance-bart0-base-ocl/continual.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
  --templates postfix=_step30_l512p_adam_lora_f8n10_continual_greedy_eval_mmlu_text-lr${LR}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --max_step 1000 --gc




  # mirpred rep
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
  configs/mmlu/shards_text.yaml configs/p3/instance-bart0-base-ocl/mirpred_flan_t5/mirpred_rep_dev_flan_t5_lora_no_repeat.yaml configs/p3/instance-bart0-base-ocl/continual.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
  --templates postfix=_step30_l512p_n8f10_continual_adam_lora_greedy_eval_mmlu_text-lr${LR}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL} task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --max_step 1000  --gc



  # mirpred thres
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
  configs/mmlu/shards_text.yaml configs/p3/instance-bart0-base-ocl/mirpred_flan_t5/mirpred_thres_dev_flan_t5_lora_fix_no_repeat.yaml configs/p3/instance-bart0-base-ocl/continual.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
  --templates postfix=_step30_l512p_n8f10_continual_adam_lora_greedy_eval_mmlu_text-lr${LR}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL} task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --max_step 1000  --gc



  # mirpred mir
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
  configs/mmlu/shards_text.yaml configs/p3/instance-bart0-base-ocl/mirpred_flan_t5/mirpred_thres_dev_flan_t5_lora_no_repeat.yaml configs/p3/instance-bart0-base-ocl/continual.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
  --templates postfix=_step30_l512p_n8f10_continual_adam_lora_greedy_eval_mmlu_text-lr${LR}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL} task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK} --max_step 1000  --gc




  echo "Created tmux session: ${session_name}"
done
done

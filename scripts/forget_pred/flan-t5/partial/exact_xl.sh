#!/bin/bash

PART=head
LR=lr1e-3
STEP=30

SHARD_IDX=0
SHARD_TOTAL=8



for task in mmlu
do

  PART="head"
  STEP=30
#  python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml \
#  configs/mmlu/shards_text.yaml \
#  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
#  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/instance-bart0-base-ocl/optims/${PART}.yaml \
#  --templates postfix=_step30_l512p_adam_partial_greedy_eval_mmlu_text-lr${LR}/${PART}/${TASK}/${SHARD_IDX}_${SHARD_TOTAL}/ task=${TASK} shard_idx=${SHARD_IDX} shard_total=${SHARD_TOTAL} --ocl_task ${TASK}
#


  python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/mmlu/fpd_mmlu_text_1e-4.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
  configs/p3/fpd/fpd_logits_defaults.yaml configs/mmlu/fpd_mmlu_text_lora_xl.yaml configs/p3/flan_mmlu.yaml configs/p3/fpd/ce_weight_0.1.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/prior_odd.yaml  \
  configs/p3/fpd/margin_rw_multi_ts.yaml configs/p3/fpd/exact.yaml \
  --templates postfix=_fpd_reps_lora_exact/head/step100k_margin_rw_sc10_ce0.1_lr1e-4_tok_small_baselr1e-4/mmlu task=mmlu PART=${PART} LR=${LR} STEP=${STEP} --ocl_task mmlu --do_train --eval_step 1000



done


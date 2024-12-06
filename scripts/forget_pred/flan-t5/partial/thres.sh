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


  python thres_predict.py --config_files configs/p3/p3_default.yaml configs/mmlu/fpd_mmlu_text_1e-4.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
  configs/p3/fpd/fpd_logits_defaults.yaml configs/mmlu/fpd_mmlu_text_partial_xl.yaml configs/p3/flan_mmlu.yaml configs/p3/fpd/ce_weight_0.1.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/prior_odd.yaml  \
  configs/p3/fpd/margin_rw_multi_ts.yaml configs/p3/fpd/exact.yaml \
  --templates postfix=_fpd_parital_lora/head/step100k_margin_rw_sc10_ce0.1_lr1e-4_tok_small_baselr1e-4/mmlu task=mmlu PART=${PART} LR=${LR} STEP=${STEP} --ocl_task mmlu



done


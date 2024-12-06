#!/bin/bash

PART=l23
LR=lr1e-3
STEP=30

SHARD_IDX=0
SHARD_TOTAL=8



for task in mmlu
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/mmlu/fpd_mmlu_text_1e-4.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/p3/fpd/fpd_logits_defaults.yaml configs/mmlu/fpd_mmlu_text_partial.yaml configs/p3/flan_mmlu_512p.yaml  configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/margin_rw.yaml  \
--templates postfix=_fpd_logits_partial/l23/step100k_margin_rw_single_sc10_lr1e-4_tok_small_baselr1e-3_fix/mmlu task=mmlu PART=${PART} LR=${LR} STEP=${STEP} --ocl_task mmlu --do_train --eval_step 1000
done

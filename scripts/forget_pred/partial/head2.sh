#!/bin/bash

PART=head
LR=lr1e-3
STEP=30

for task in super_glue-cb super_glue-copa super_glue-rte
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/prior_odd.yaml configs/p3/fpd/bart_text_partial.yaml \
--templates postfix=_fpd_partial_paired_mean_mlp_prior_odd/head/step100k/${task} task=${task} PART=${PART} LR=${LR} STEP=${STEP} --ocl_task ${task} --do_train --eval_step 1000
done
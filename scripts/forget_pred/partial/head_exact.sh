#!/bin/bash

PART=head
LR=lr1e-3
STEP=30

for task in super_glue-cb #super_glue-copa super_glue-rte super_glue-wsc.fixed super_glue-wic anli winogrande-winogrande_xl hellaswag
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/bart_text_partial.yaml \
configs/p3/fpd/multi_token_kl.yaml \
configs/p3/fpd/margin_rw_multi_ts.yaml \
configs/p3/fpd/exact.yaml \
--templates postfix=_fpd_partial_logits_exact_surface/head/step100k/${task} task=${task} PART=${PART} LR=${LR} STEP=${STEP} --ocl_task ${task} --do_eval --eval_step 1000 --return_pred_logits
done
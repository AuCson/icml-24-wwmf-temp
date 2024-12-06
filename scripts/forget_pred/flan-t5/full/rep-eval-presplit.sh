#!/bin/bash

python train_logit_predict.py --config_files configs/p3/p3_default.yaml \
configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml \
configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text_full_presplit.yaml configs/p3/fpd/lr1e-4.yaml \
configs/p3/fpd/ce_weight_0.1.yaml configs/p3/fpd/lr_scale10.yaml configs/p3/fpd/step100k.yaml \
--templates postfix=_fpd_rep_multi_full/mmlu task=mmlu --ocl_task mmlu --eval_step 1000 --do_eval
#!/bin/bash

python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
configs/p3/flan_mmlu_512p.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml \
configs/mmlu/fpd_mmlu_text.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml \
configs/p3/fpd/logit_direct_ce.yaml configs/p3/fpd/ckpt_1k.yaml --eval_step 1000 --templates postfix=_fpd_flan_direct_ce_1k/ task=mmlu --ocl_task mmlu \
--do_train
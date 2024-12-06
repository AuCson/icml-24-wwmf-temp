#!/bin/bash

LR=1e-6


#for task in mmlu
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_logits_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr2e-5.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/margin_rw_multi_ts.yaml configs/p3/fpd/step100k.yaml  \
#--templates postfix=_fpd_logits_multi/step100k_margin_rw_lr2e-5_sc10_fix_tok_small/mmlu task=mmlu --ocl_task mmlu --do_train --eval_step 1000
#done

for task in mmlu
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/p3/fpd/fpd_logits_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/margin_value_0.1.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/margin_rw_multi_ts.yaml  \
--templates postfix=_fpd_logits_multi/step100k_margin_rw_lr1e-4_sc10_mv0.1_tok_small/mmlu task=mmlu --ocl_task mmlu --do_train --eval_step 1000
done
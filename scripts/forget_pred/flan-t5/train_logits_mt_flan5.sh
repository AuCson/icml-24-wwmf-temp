#!/bin/bash

LR=1e-6


#for task in mmlu
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_logits_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr2e-5.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/margin_rw_multi_ts.yaml configs/p3/fpd/step100k.yaml \
#configs/p3/fpd/margin_loss_0.01.yaml \
#--templates postfix=_fpd_logits_multi/step100k_margin_rw0.01_lr2e-5_sc10_fix_tok_small/mmlu task=mmlu --ocl_task mmlu --do_train --eval_step 1000
#done

for task in mmlu
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/fpd_rep_task_bin.yaml configs/mmlu/fpd_olmo_instruct_presplit.yaml \
configs/p3/fpd/lr1e-5.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/lr_scale10.yaml   configs/p3/fpd/accum8.yaml \
--templates postfix=_fpd_olmo/rep-based-bin-1e-5-ce1-accum8 task=mmlu --ocl_task mmlu --do_train --eval_step 2000 --skip_first_eval

done
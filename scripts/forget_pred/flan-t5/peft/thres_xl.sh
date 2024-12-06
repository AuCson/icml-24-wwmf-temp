#!/bin/bash

PART=head
LR=lr1e-3
STEP=30


#for task in mmlu
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/mmlu/fpd_mmlu_text_1e-4.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text_lora.yaml configs/p3/flan_mmlu.yaml configs/p3/fpd/ce_weight_0.1.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/prior_odd.yaml  \
#--templates postfix=_fpd_reps_lora_prior_odd/step100k_margin_rw_sc10_ce0.1_lr1e-4_tok_small_baselr1e-3/mmlu task=mmlu PART=${PART} LR=${LR} STEP=${STEP} --ocl_task mmlu --do_train --eval_step 1000
#done


for task in mmlu
do
python thres_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text_lora_xl.yaml configs/p3/flan_mmlu_512p.yaml configs/p3/fpd/ce_weight_0.1.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/prior_odd.yaml  \
--templates postfix=_fpd_reps_lora_thres/mmlu task=mmlu PART=${PART} LR=${LR} STEP=${STEP} --ocl_task mmlu
done



#--config_files configs/p3/p3_default.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/mmlu/fpd_mmlu_text_partial.yaml configs/p3/flan_mmlu.yaml
#configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml
#configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml
#configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml
#--templates postfix=_step100_l512_adam_partial_greedy_eval_mmlu_text-lr1e-3/head/mmlu/0_8 task=mmlu PART=head LR=lr1e-3 STEP=100 --ocl_task mmlu
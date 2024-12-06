#!/bin/bash

#!/bin/bash

LR=1e-6

#for task in super_glue-wic hellaswag anli winogrande-winogrande_xl
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
#configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/prior_odd.yaml configs/p3/fpd/ckpt_2k.yaml --templates postfix=_fpd_paired_mean_mlp/step10k_bias_odd/${task} task=${task} --ocl_task ${task} --do_train
#done

#for task in mmlu
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml \
#--templates postfix=_fpd_rep_multi/step100k_lr1e-4_sc10_tok_small/mmlu task=mmlu --ocl_task mmlu --do_train --eval_step 1000
#done

#for task in mmlu
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml \
#--templates postfix=_fpd_rep_multi/step100k_lr1e-4_sc10_tok_small/mmlu task=mmlu --ocl_task mmlu --do_train --eval_step 1000
#done

for task in mmlu
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/mmlu/fpd_mmlu_text_1e-4.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/ce_weight_0.1.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml \
--templates postfix=_fpd_rep_multi/step100k_lr1e-4_ce0.1_sc10_tok_small_baselr1e-4/mmlu task=mmlu --ocl_task mmlu --do_train --eval_step 1000
done
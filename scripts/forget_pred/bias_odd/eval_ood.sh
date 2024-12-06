#!/bin/bash

LR=1e-6

for src_task in super_glue-cb super_glue-copa super_glue-rte super_glue-wsc.fixed super_glue-wic hellaswag anli winogrande-winogrande_xl
do
for task in super_glue-wic winogrande-winogrande_xl # super_glue-cb super_glue-copa super_glue-rte super_glue-wsc.fixed anli hellaswag
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/load_model_dir.yaml configs/p3/fpd/prior_odd.yaml --templates postfix=_fpd_paired_mean_mlp_ood_eval/step10k_bias_odd/src_${src_task}/${task} load_model_dir=runs/instance-p3-bart0-large/vanilla_bg100_fpd_paired_mean_mlp/step10k_bias_odd/${src_task} task=${task} --ocl_task ${task} --do_eval

python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/load_model_dir.yaml --templates postfix=_fpd_paired_mean_mlp_ood_eval/step10k/src_${src_task}/${task} load_model_dir=runs/instance-p3-bart0-large/vanilla_bg100_fpd_paired_mean_mlp/step10k_bias_odd/${src_task} task=${task} --ocl_task ${task} --do_eval


done
done





#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
 #configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/lr_scale100.yaml --templates postfix=_fpd_paired_mean_mlp/step10k_ls100/super_glue-cb task=super_glue-cb --ocl_task super_glue-cb --do_train

#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
 #configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/lr1e-5.yaml --templates postfix=_fpd_paired_mean_mlp/step10k_lr1e-5/super_glue-cb task=super_glue-cb --ocl_task super_glue-cb --do_train

#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
#configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/temp0.1.yaml --templates postfix=_fpd_paired_mean_mlp/step10k_lr1e-5/super_glue-cb task=super_glue-cb --ocl_task super_glue-cb --do_train
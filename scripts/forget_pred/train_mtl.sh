#!/bin/bash

LR=1e-6

for mtl_tasks in super_glue-cb+super_glue-copa+super_glue-rte super_glue-cb+super_glue-
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/mtl.yaml --templates postfix=_fpd_paired_mean_mlp_mtl/step10k/{task} mtl_tasks=${mtl_tasks} --ocl_task debug --do_train
done


#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
 #configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/lr_scale100.yaml --templates postfix=_fpd_paired_mean_mlp/step10k_ls100/super_glue-cb task=super_glue-cb --ocl_task super_glue-cb --do_train

#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
 #configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/lr1e-5.yaml --templates postfix=_fpd_paired_mean_mlp/step10k_lr1e-5/super_glue-cb task=super_glue-cb --ocl_task super_glue-cb --do_train

#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
#configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/temp0.1.yaml --templates postfix=_fpd_paired_mean_mlp/step10k_lr1e-5/super_glue-cb task=super_glue-cb --ocl_task super_glue-cb --do_train
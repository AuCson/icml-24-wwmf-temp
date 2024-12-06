#!/bin/bash

LR=1e-6

#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
#configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml --templates postfix=_fpd_paired_mean_mlp/step50k/super_glue-cb task=super_glue-cb --ocl_task super_glue-cb --do_train

#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
 #configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/lr_scale100.yaml --templates postfix=_fpd_paired_mean_mlp/step10k_ls100/super_glue-cb task=super_glue-cb --ocl_task super_glue-cb --do_train

#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
 #configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/lr1e-5.yaml --templates postfix=_fpd_paired_mean_mlp/step10k_lr1e-5/super_glue-cb task=super_glue-cb --ocl_task super_glue-cb --do_train



#for task in super_glue-wsc.fixed
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
#configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/p3/fpd/multi_token.yaml --templates postfix=_fpd_logits_multi/step10k/${task} task=${task} --ocl_task ${task} --do_eval
#done

#for src_task in super_glue-wsc.fixed anli
#do
#for task in super_glue-cb super_glue-copa super_glue-rte super_glue-wsc.fixed super_glue-wic anli hellaswag winogrande-winogrande_xl
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
#configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/margin_rw.yaml configs/p3/fpd/load_model_dir.yaml --templates postfix=_fpd_logits_multi_kl_ood/step100k_margin_rw_tok/${task} task=${task} \
#--ocl_task ${task}  --do_eval --eval_step 1000 --return_pred_logits --load_model_dir runs/instance-p3-bart0-large/vanilla_bg100_fpd_logits_multi_kl/step100k_margin_rw_tok/${src_task}
#done
#done

for src_task in super_glue-cb super_glue-copa super_glue-rte
do
for task in super_glue-cb super_glue-copa super_glue-rte
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/margin_rw_multi_ts.yaml configs/p3/fpd/ds_offset100.yaml configs/p3/fpd/load_model_dir.yaml --templates postfix=_fpd_logits_multi_kl_offset100/step100k_margin_rw_mtts_sc10_tok/${src_task}/${task} task=${task} \
--ocl_task ${task}  --do_eval --eval_step 1000 --return_pred_logits --load_model_dir runs/instance-p3-bart0-large/vanilla_bg100_fpd_logits_multi_kl/step100k_margin_rw_mtts_sc10_tok/${src_task}

#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
#configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/ds_offset100.yaml --templates postfix=_fpd_paired_mean_mlp/step10k/${task}  task=${task} --ocl_task ${task} --do_eval

done
done
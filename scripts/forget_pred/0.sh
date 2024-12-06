##!/bin/bash
#
#
#CONFIG=vanilla
#GPU_TYPE=6000
#
#echo "greedy decoding and eval model, 30step, sgd"
#
#for TASK in super_glue-cb super_glue-copa super_glue-rte #super_glue-wic super_glue-wsc.fixed winogrande-winogrande_xl anli hellaswag
#do
#
#  python logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml --templates postfix=_lr1e-6_step30_greedy_eval_fixbos/${TASK} task=${TASK} --ocl_task ${TASK} --exp_name simple-0.2 --coeff 0.2 --optimizer adam
#  #python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/subset.yaml --templates postfix=_adam_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#  #python logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml  configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_lr1e-6_step30_greedy_eval_fixbos/${TASK} --exp_name simple-0.2 --ocl_task ${TASK} --coeff 0.2 --optimizer adam
#
#done

for task in super_glue-cb super_glue-copa super_glue-rte super_glue-wic super_glue-wsc.fixed winogrande-winogrande_xl anli hellaswag
do
 python scripts/fpd_pred_split.py   /home/xsjin/cl-analysis/runs/instance-p3-bart0-large/vanilla_bg100_step30_adam_partial_greedy_eval_fixbos-lr1e-3/head/${task}
done

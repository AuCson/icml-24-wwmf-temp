#!/bin/bash


LR=1e-6
STEP=30

for task in super_glue-cb super_glue-copa super_glue-rte super_glue-wsc.fixed super_glue-wic anli winogrande-winogrande_xl hellaswag
do

python thres_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml \
configs/p3/instance-bart0-base-ocl/lr${LR}.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/p3/fpd/multi_token_kl.yaml \
configs/p3/fpd/step100k.yaml configs/p3/fpd/margin_rw_multi_ts.yaml \
--templates postfix=_fpd_thres/${task} task=${task} LR=lr1e-6 STEP=30 \
--ocl_task ${task}

done
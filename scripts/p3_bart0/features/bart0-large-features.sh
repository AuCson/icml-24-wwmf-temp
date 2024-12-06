#!/bin/bash

LR=1e-2
CONFIG=vanilla
echo "step is 30, sgd, greedy, fixbos"

for task in super_glue-cb
do
python extract_features.py --type pt --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/${CONFIG}_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml  --templates postfix=_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${task} --ocl_task ${task}
done

for task in super_glue-cb super_glue-copa super_glue-rte super_glue-wsc.fixed winogrande-winogrande_xl anli hellaswag super_glue-wic
do
python extract_features.py --type ocl --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/${CONFIG}_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml  --templates postfix=_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${task} --ocl_task ${task}
done
#!/bin/bash

LR=1e-2
CONFIG=vanilla
echo "step is 30"
for TASK in super_glue-cb super_glue-copa super_glue-rte
do
#python stat_logits.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/${CONFIG}_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml  --templates postfix=_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${task} --ocl_task ${task}

  # mir-pred
  python stat_grads.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_lr1e-6_step30_greedy_eval_fixbos/${TASK} task=${TASK} --ocl_task ${TASK} --type prod --exp_name clip.adam


done
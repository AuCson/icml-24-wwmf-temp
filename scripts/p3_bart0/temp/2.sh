#!/bin/bash

CONFIG=vanilla
GPU_TYPE=2080

echo "greedy decoding and eval model, 50step, sgd"

for TASK in winogrande-winogrande_xl hellaswag
do
  for LR in 1e-2
  do
python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/${CONFIG}_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps50.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml  --templates postfix=_lr1e-6_step50_sgd_greedy_eval_fixbos_loss-lr${LR}/${TASK} --ocl_task ${TASK}

done
done
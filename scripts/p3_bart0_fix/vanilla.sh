#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

TASKS=${1}

LR=1e-6
echo "adam with LR ${LR}"

for TASK in ${TASKS}

do

  #python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_step30_adam-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/fix_decoder_start_token_id.yaml --templates postfix=_step30_adam_fixbos_fixdst-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}

done
#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, adam"

TASKS=${1}


for TASK in super_glue-cb super_glue-copa super_glue-rte super_glue-wsc.fixed super_glue-wic hellaswag anli winogrande-winogrande_xl

do
  for LR in 1e-6
  do
  # vanilla
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml \
   configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/ocl_dev.yaml configs/p3/instance-bart0-base-ocl/mir256_distill.yaml \
   configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}_fix.yaml \
   configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml \
   configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_step30_adam_full_continual_dev_greedy_eval_fixbos-lr${LR}_fix/${TASK} \
   task=${TASK} --ocl_task ${TASK} --max_step 100



  echo "Created tmux session: ${session_name}"
done
done
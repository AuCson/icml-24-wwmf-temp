#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

TASKS=${1}

LR=1e-6
echo "adam, distill-er with LR ${LR}"

for TASK in ${TASKS}

do

  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/distill_er.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_subset_step30_adam-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
done
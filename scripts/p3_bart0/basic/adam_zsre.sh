#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, adam"

#TASKS=${1}



# vanilla
python instance_ocl_p3.py --config_files configs/p3/zsre_nq.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg10k_bart_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/replay_n.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_step30_adam_full_greedy_eval_fixbos-lr1e-6/${TASK} task=zsre --ocl_task zsre --max_step 500

# vanilla
python instance_ocl_p3.py --config_files configs/p3/zsre_nq.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg10k_bart_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/distill_er_zsre.yaml configs/p3/instance-bart0-base-ocl/replay_n.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_step30_adam_full_greedy_eval_fixbos-lr1e-6/${TASK} task=zsre --ocl_task zsre --max_step 500




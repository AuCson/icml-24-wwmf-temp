#!/bin/bash


CONFIG=vanilla
GPU_TYPE=8000

echo "greedy decoding and eval model, 30step, sgd"

for TASK in super_glue-cb #super_glue-copa super_glue-rte
do
  for LR in 1e-2
  do
  session_name=OCL_${TASK}_${CONFIG}_SGD${LR}_large-mirpred
  session_name="${session_name/\./-}"
  #python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-2.yaml configs/p3/instance-bart0-base-ocl/mir1k.yaml configs/p3/instance-bart0-base-ocl/replay_freq.yaml configs/p3/instance-bart0-base-ocl/replay_n.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml --templates postfix=_freq1_n32_nstep4_lr1e-6_step30_sgd_greedy_eval_fixbos_subset-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}

  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-2.yaml configs/p3/instance-bart0-base-ocl/mir_pred.yaml configs/p3/instance-bart0-base-ocl/replay_freq.yaml configs/p3/instance-bart0-base-ocl/replay_n.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml --templates postfix=_freq1_n32_nstep4_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}


  #python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/insmtance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-2.yaml configs/p3/instance-bart0-base-ocl/vanilla.yaml --templates postfix=_lr1e-6_step30_sgd_greedy_eval_fixbos_subset-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}


  echo "Created tmux session: ${session_name}"
done
done
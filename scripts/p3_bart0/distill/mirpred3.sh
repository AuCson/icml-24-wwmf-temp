#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, adam"

#TASKS=${1}

for TASK in super_glue-cb super_glue-copa super_glue-rte super_glue-wsc.fixed super_glue-wic anli winogrande-winogrande_xl hellaswag

do
  for LR in 1e-6
  do

  # mir-pred, thres
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/mirpred_subset_distill_adam.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/mirpred_rep_dev_mtl.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_delay_lr1e-6_step30_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}


  # mir-pred, thres ss
  #python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/mirpred_subset_distill_adam.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/mirpred_rep_dev_subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_delay_lr1e-6_step30_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}



  echo "Created tmux session: ${session_name}"
done
done



##!/bin/bash
#
#
#CONFIG=vanilla
#GPU_TYPE=6000
#
#echo "greedy decoding and eval model, 30step, adam"
#
##TASKS=${1}
#
#
#for TASK in winogrande-winogrande_xl
#
#do
#  for LR in 1e-6
#  do
#  # er-distill
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/distill_er.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_adam_delay_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#  # mir1k-distill
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/mir1k_distill.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_adam_delay_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#  # mir-pred
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/mirpred_subset_distill_adam.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_delay_0.2_lr1e-6_step30_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#  # mir-pred-ref
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/mirpred_subset_distill_adam_ref.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_delay_ref0.2_lr1e-6_step30_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#
#  echo "Created tmux session: ${session_name}"
#done
#done
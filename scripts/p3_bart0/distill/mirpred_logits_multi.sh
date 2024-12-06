#!/bin/bash


CONFIG=vanilla
GPU_TYPE=6000

echo "greedy decoding and eval model, 30step, sgd"

#TASKS=${1}

LR=1e-6

for TASK in super_glue-wsc.fixed super_glue-wic winogrande-winogrande_xl anli hellaswag super_glue-cb super_glue-copa super_glue-rte
do
   #mir-pred
  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml \
  configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml \
  configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/mirpred_subset_distill_adam.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
  configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/mirpred_logit_margin_dev_mtts.yaml \
  configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_new_freq10_n8_nstep4_delay_lr1e-6_step30_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}

done


#for TASK in anli hellaswag
#
#do
#  for LR in 1e-6
#  do
#  # vanilla
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_step30_adam_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#
#  # er-distill, nstep1
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/distill_er.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep1_adam_delay_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#  # er-distill, nstep16
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/distill_er.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep16.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep16_adam_delay_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#
#  # mir1k-distill
#  # python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr${LR}.yaml configs/p3/instance-bart0-base-ocl/mir1k_distill.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_adam_delay_lr1e-6_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#  # mir-pred
#  #python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/mirpred_subset_distill_adam.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_delay_0.2_lr1e-6_step30_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#  # mir-pred-ref
#  #python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/mirpred_subset_distill_adam_ref.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq10_n8_nstep4_delay_ref0.2_lr1e-6_step30_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#
#  echo "Created tmux session: ${session_name}"
#done
#done


##!/bin/bash
#
#
#CONFIG=vanilla
#GPU_TYPE=6000
#
#echo "greedy decoding and eval model, 30step, sgd"
#
#TASKS=${1}
#
#for TASK in ${TASKS}
#do
#  echo ${TASK}
#  for LR in 1e-2
#  do
#  # vanilla
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-2.yaml configs/p3/instance-bart0-base-ocl/replay_n.yaml configs/p3/instance-bart0-base-ocl/replay_freq.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_lr1e-6_step30_sgd_greedy_eval_fixbos_subset-lr{LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#done
#
#
#for TASK in ${TASKS}
#
#do
#  for LR in 1e-2
#  do
#  # er
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-2.yaml configs/p3/instance-bart0-base-ocl/distill_er.yaml configs/p3/instance-bart0-base-ocl/replay_n.yaml configs/p3/instance-bart0-base-ocl/replay_freq.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq1_n32_nstep4_delay_sep_lr1e-2_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#  # mir1k
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-2.yaml configs/p3/instance-bart0-base-ocl/mir1k.yaml configs/p3/instance-bart0-base-ocl/replay_n.yaml configs/p3/instance-bart0-base-ocl/replay_freq.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/subset.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml --templates postfix=_freq1_n32_nstep4_delay_sep_lr1e-2_step30_sgd_greedy_eval_fixbos-lr${LR}/${TASK} task=${TASK} --ocl_task ${TASK}
#
#
#  echo "Created tmux session: ${session_name}"
#done


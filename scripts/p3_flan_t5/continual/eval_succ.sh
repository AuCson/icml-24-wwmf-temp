#!/bin/bash

#python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml \
# configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml configs/mmlu/shards_text_xl_dev.yaml \
# configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
# configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
# configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
# configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
# --templates postfix=_mmlu_succ_eval/0_8/ \
# task=${TASK} shard_idx=0 shard_total=8 --ocl_task mmlu --max_step 1000  \
# --load_base_ckpt runs/instance-p3-flan-t5-xl/vanilla_bg100_step30_l512p_adam_continualfix_lora_greedy_eval_mmlu_text-lr1e-4/mmlu/0_8/continual_model.pt \
# --eval_ocl_only

# python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml \
# configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml configs/mmlu/shards_text_xl_dev.yaml \
# configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
# configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
# configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
# configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
# --templates postfix=_mmlu_succ_eval/0_8/ \
# task=${TASK} shard_idx=0 shard_total=8 --ocl_task mmlu --max_step 1000  \
# --load_base_ckpt runs/instance-p3-flan-t5-xl/mir_pred_rep_prior_odd_no_repeat__step30_l512p_n2f5_adam_continualfix_lora_greedy_eval_mmlu_text-lr1e-4/mmlu/0_8/continual_model.pt \
# --eval_ocl_only
#
# python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml \
# configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml configs/mmlu/shards_text_xl_dev.yaml \
# configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
# configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
# configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
# configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
# --templates postfix=_mmlu_succ_eval/0_8/ \
# task=${TASK} shard_idx=0 shard_total=8 --ocl_task mmlu --max_step 1000  \
# --load_base_ckpt runs/instance-p3-flan-t5-xl/mir_pred_gt_no_repeat__step30_l512p_n2f5_adam_continualfix_lora_greedy_eval_mmlu_text-lr1e-4/mmlu/0_8/continual_model.pt \
# --eval_ocl_only
#
# python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml \
# configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml configs/mmlu/shards_text_xl_dev.yaml \
# configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
# configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
# configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
# configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
# --templates postfix=_mmlu_succ_eval/0_8/ \
# task=${TASK} shard_idx=0 shard_total=8 --ocl_task mmlu --max_step 1000  \
# --load_base_ckpt runs/instance-p3-flan-t5-xl/mir_pred_thres_no_repeat__step30_l512p_n2f5_adam_continualfix_lora_greedy_eval_mmlu_text-lr1e-4/mmlu/0_8/continual_model.pt \
# --eval_ocl_only
#
#  python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml \
# configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml configs/mmlu/shards_text_xl_dev.yaml \
# configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
# configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
# configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
# configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
# --templates postfix=_mmlu_succ_eval/0_8/ \
# task=${TASK} shard_idx=0 shard_total=8 --ocl_task mmlu --max_step 1000  \
# --load_base_ckpt runs/instance-p3-flan-t5-large/mir1k_64_distill_er_star_step30_l512p_adam_continual_lora_f2n5_greedy_eval_mmlu_text-lr1e-4/mmlu/0_8/continual_model.pt \
# --eval_ocl_only

   python instance_ocl_p3.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml \
 configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml configs/mmlu/shards_text_xl_dev.yaml \
 configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
 configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
 configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
 configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/flan_t5_lora.yaml \
 --templates postfix=_mmlu_succ_eval/0_8/ \
 task=${TASK} shard_idx=0 shard_total=8 --ocl_task mmlu --max_step 1000  \
 --load_base_ckpt runs/instance-p3-flan-t5-large/distill_er_coreset_step30_l512p_adam_continual_lora_f2n5_greedy_eval_mmlu_text-lr1e-4/mmlu/0_8/continual_model.pt \
 --eval_ocl_only
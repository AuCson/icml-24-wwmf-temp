#!/bin/bash

replay_every=8

for replay_every in 8 32
do
for task_id in 65 18 8 60 6
do
  for temp in 10.0
  do
  config="temp_${temp}_re${replay_every}"
  python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
 configs/llm/ocl/7b_peft_flan_5kstep.yaml configs/llm/ocl/mir_pred_dolma_flan.yaml configs/llm/ocl/mir_pred_temp.yaml \
 configs/p3/instance-bart0-base-ocl/greedy.yaml configs/llm/ocl/replay_every.yaml \
 configs/p3/instance-bart0-base-ocl/lr1e-4.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
 --ocl_task flan --monitor lm_loss --n_gpu 1 --skip_before_eval --is_ocl_training_only --templates TASK_ID=${task_id} \
 CONFIG=${config} weight_temp=${temp} replay_freq=${replay_every}

  python vllm_exps.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
  configs/llm/stats/7b_flan_dolma_tokenize_fix_cl.yaml --templates ocl_task_id=${task_id} \
  task_model_dir="runs_olmo_ocl/flan-5k-mirpred-gt-${config}/task_${task_id}/model_save" \
  cl_method="mirpred-gt-${config}" --stat_ppl --skip_eval_ocl_ds
  done
done
done

#task_id=6
#for task_id in 6 65 18
#do
#for temp in 1.0 0.3
#do
#  config="temp_${temp}"
#  python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
# configs/llm/ocl/7b_peft_flan_5kstep.yaml configs/llm/ocl/mir_pred_dolma_flan.yaml configs/llm/ocl/mir_pred_temp.yaml \
# configs/p3/instance-bart0-base-ocl/greedy.yaml \
# configs/p3/instance-bart0-base-ocl/lr1e-4.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
# --ocl_task flan --monitor lm_loss --n_gpu 1 --skip_before_eval --is_ocl_training_only --templates TASK_ID=${task_id} \
# CONFIG=${config} weight_temp=${temp}
#
#  python vllm_exps.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
#configs/llm/stats/7b_flan_dolma_tokenize_fix_cl.yaml --templates ocl_task_id=${task_id} \
#task_model_dir="runs_olmo_ocl/flan-5k-mirpred-gt-${config}/task_${task_id}/model_save" \
#cl_method="mirpred-gt-${config}" --stat_ppl --skip_eval_ocl_ds
#
#done
#done
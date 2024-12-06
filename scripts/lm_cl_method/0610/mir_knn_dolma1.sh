#!/bin/bash

replay_every=8

for task_id in 6 65 18
do
  for temp in 0.2
  do
  config="temp_${temp}_re${replay_every}"
  python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
 configs/llm/ocl/7b_peft_flan_5kstep.yaml configs/llm/ocl/mir_pred_dolma_flan_knn.yaml configs/llm/ocl/mir_pred_temp.yaml \
 configs/p3/instance-bart0-base-ocl/greedy.yaml configs/llm/ocl/replay_every.yaml \
 configs/p3/instance-bart0-base-ocl/lr1e-4.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
 --ocl_task flan --monitor lm_loss --n_gpu 1 --skip_before_eval --is_ocl_training_only --templates TASK_ID=${task_id} \
 CONFIG=${config} weight_temp=${temp} replay_freq=${replay_every}

  python vllm_exps.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
  configs/llm/stats/7b_flan_dolma_tokenize_fix_cl.yaml --templates ocl_task_id=${task_id} \
  task_model_dir="runs_olmo_ocl/flan-5k-mirpred-knn-${config}/task_${task_id}/model_save" \
  cl_method="mirpred-knn-${config}" --stat_ppl --skip_eval_ocl_ds
  done
done


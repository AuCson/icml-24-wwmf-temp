#!/bin/bash


task_id=0
python vllm_exps.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
configs/llm/stats/7b_flan_dolma_tokenize_fix.yaml --templates ocl_task_id=${task_id} --stat_ppl --skip_eval_ocl_ds --eval_base

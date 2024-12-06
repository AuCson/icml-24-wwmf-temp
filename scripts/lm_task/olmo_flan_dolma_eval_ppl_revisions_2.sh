#!/bin/bash

revision=2
task_id=0
echo "Eval base model"
python vllm_exps.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
configs/llm/stats/7b_flan_dolma_tokenize_fix_revision.yaml --templates ocl_task_id=${task_id} revision=${revision} \
--stat_ppl --skip_eval_ocl_ds --eval_base



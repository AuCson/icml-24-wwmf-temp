#!/bin/bash

start_task_id=${1}
stop_task_id=${2}

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Tulu eval: Current task id ${task_id}"


python vllm_exps.py  --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml \
configs/llm/stats/7b_mmlu_tulu.yaml --templates ocl_task_id=${task_id} --stat_output --stat_ppl --ocl_split test


done
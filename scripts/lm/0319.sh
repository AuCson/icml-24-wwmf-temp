#!/bin/bash

python stream_ocl.py --config_files configs/p3/p3_default.yaml \
configs/llm/dolma_defaults.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
configs/p3/instance-bart0-base-ocl/lr1e-6_fix.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/llm/paloma_v1/0319.yaml \
--ocl_task paloma --monitor lm_loss --n_gpu 2 --skip_before_eval --is_lm_sep_task_exp \
--start_task_id ${1} --stop_task_id ${2} \
--templates shard=${3}


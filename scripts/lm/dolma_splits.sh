#!/bin/bash

minn=${1}
maxx=${2}

for ((start = minn ; start < maxx ; start++ ))
do

stop=$((start + 1))

echo ${start}
echo ${stop}

python stream_ocl.py --config_files configs/p3/p3_default.yaml \
configs/llm/dolma_defaults.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
configs/p3/instance-bart0-base-ocl/lr1e-6_fix.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/llm/paloma_v1/dolma_splits.yaml \
--ocl_task paloma --monitor lm_loss --n_gpu 2 --is_lm_sep_task_exp \
--start_task_id 0 --stop_task_id 1 \
--before_eval_only \
--templates start=${start} stop=${stop}

done
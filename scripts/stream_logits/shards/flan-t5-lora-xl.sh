#!/bin/bash

for shard_idx in 0 1 2 3 4 5 6 7
do
python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml \
configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl_analysis.yaml configs/mmlu/shards_text_xl.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml \
configs/p3/instance-bart0-base-ocl/lr1e-4.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/stream.yaml configs/p3/flan_t5_lora.yaml \
--templates postfix=flan-t5-xl-lora-lr1e-4-wpred/shard_${shard_idx} task=mmlu shard_idx=${shard_idx} shard_total=8 \
--ocl_task mmlu --max_step 20 --skip_before_eval --monitor logit preds
done
#!/bin/bash

#python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml \
#configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl_analysis.yaml configs/mmlu/shards_text.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml \
#configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
#configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml \
#configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/stream.yaml configs/p3/flan_t5_lora.yaml \
#--templates postfix=flan-t5-xl-lora task=mmlu shard_idx=0 shard_total=8 --ocl_task mmlu --max_step 100 --skip_before_eval --monitor logit


python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/p3/counterfact.yaml \
configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl_analysis.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml \
configs/p3/instance-bart0-base-ocl/lr1e-4.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml \
configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/stream.yaml configs/p3/flan_t5_lora.yaml configs/counterfact/episodic.yaml \
--templates postfix=flan-t5-lora-xl-ft-1e-4 task=mmlu shard_idx=0 shard_total=8 --ocl_task mmlu --max_step 10000 --monitor logit
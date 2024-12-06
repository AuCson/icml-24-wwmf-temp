#!/bin/bash

python coreset.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml \
configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml configs/mmlu/shards_text_dev.yaml \
configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
--templates postfix=_coreset task=mmlu shard_idx=0

python coreset.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml \
configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_xl.yaml configs/mmlu/shards_text_dev.yaml \
configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
--templates postfix=_coreset task=mmlu shard_idx=0

python coreset.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml \
configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/mmlu/shards_text_dev.yaml \
configs/p3/instance-bart0-base-ocl/continual.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
configs/p3/instance-bart0-base-ocl/replay_n8.yaml configs/p3/instance-bart0-base-ocl/replay_freq10.yaml \
configs/p3/instance-bart0-base-ocl/replay_nstep.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
--templates postfix=_coreset task=mmlu shard_idx=0

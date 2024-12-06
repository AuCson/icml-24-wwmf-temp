#!/bin/bash

python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml configs/llm/tmp.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml \
configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
--ocl_task paloma --max_step 100 --monitor lm_loss --n_gpu 1
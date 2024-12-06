#!/bin/bash

for task in super_glue-cb super_glue-copa super_glue-rte
do
python stat_logits.py --update --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml --templates postfix=_lr1e-6_step30_greedy/${task} --ocl_task ${task}
done
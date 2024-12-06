#!/bin/bash

#start_task_id=${1}
#stop_task_id=${2}

#for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
#do
#echo "Current task id ${task_id}"
#python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml configs/llm/ocl/7b_inst_peft_truthful.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/lr1e-4.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#--ocl_task truthful --monitor lm_loss --n_gpu 1 --skip_before_eval --is_ocl_training_only --templates TASK_ID=${task_id}
#done


for task_id in 2 7
do
echo "Current task id ${task_id}"
python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/llm/dolma_defaults.yaml configs/llm/ocl/7b_inst_peft_truthful.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/lr1e-4.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
--ocl_task truthful --monitor lm_loss --n_gpu 1 --skip_before_eval --is_ocl_training_only --templates TASK_ID=${task_id}
done
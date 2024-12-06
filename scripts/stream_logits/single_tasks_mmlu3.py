import os
import yaml

with open('configs/p3/flan_mmlu_512p.yaml') as f:
    cfg = yaml.safe_load(f)
ocl_tasks = cfg['ocl_tasks']

for task in ocl_tasks[40:]:
    os.chdir("/home/xsjin/cl-analysis")
    print(task)
    os.system(
        "python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml \
        configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large_analysis.yaml \
        configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml \
        configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/replay_n8.yaml \
        configs/p3/instance-bart0-base-ocl/replay_freq10.yaml configs/p3/instance-bart0-base-ocl/replay_nstep.yaml \
        configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/stream.yaml configs/stream/stream_subset_mmlu_single.yaml \
        --templates postfix=flan-t5-large-ft-single-mmlu/{task} task={task} --ocl_task {task} --start_step 5 --max_step 20 --monitor logit".format(task=task)
    )
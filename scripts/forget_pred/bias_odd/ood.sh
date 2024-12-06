#!/bin/bash

for task in bbh
do
python thres_predict.py --config_files configs/p3/p3_default.yaml configs/p3/bbh_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_bbh.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml \
configs/p3/fpd/load_model_dir.yaml \
--templates postfix=_fpd_reps_multi_odd/step100k_margin_rw0.3_lr1e-4_sc10_tok_small/bbh task=bbh --ocl_task bbh
done

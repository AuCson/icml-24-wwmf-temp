#!/bin/bash

#!/bin/bash

LR=1e-6


for task in mmlu
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale1.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml \
configs/p3/fpd/ce_weight_0.03.yaml \
--templates postfix=_fpd_rep_multi/step100k_lr1e-4_sc1_ce0.03_tok_small/mmlu task=mmlu --ocl_task mmlu --do_train --eval_step 1000
done
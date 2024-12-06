#!/bin/bash

python scripts/fpd_split_table.py --pt_ds runs/instance-p3-flan-t5-large/vanilla_bg100_step30_adam_full_greedy_eval_mmlu_text-lr1e-6/mmlu/0_8/concat_pt_ds.csv \
--ocl_ds /home/xsjin/cl-analysis/runs/instance-p3-flan-t5-large/vanilla_bg100_step30_adam_full_greedy_eval_mmlu_text-lr1e-6/mmlu/ocl_error_ds.csv \
--forget_mat cf_cache/norm_df_fgt-t5l-full.npy \
--base_correct cf_cache/t5l-seed50-bc.pkl \
--output_file cf_cache/t5l-seed50-full-fpd-split.pkl
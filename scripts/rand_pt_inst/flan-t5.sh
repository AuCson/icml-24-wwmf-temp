#!/bin/bash

echo "Bash version ${BASH_VERSION}..."

minx=${1}
maxx=${2}

for (( seed = ${minx}; seed < ${maxx}; ++seed ))
do
  echo "seed ${seed}"

  python stream_ocl.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu_512p.yaml \
  configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml \
  configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml \
  configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/rand/inst/flan-t5-large-full.yaml \
  --templates my_seed=${seed} --ocl_task mmlu --monitor logit_pred --base_corr_only

done
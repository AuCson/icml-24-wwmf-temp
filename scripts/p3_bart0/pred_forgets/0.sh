#!/bin/bash

#for TASK in super_glue-cb super_glue-copa super_glue-rte super_glue-wsc.fixed winogrande-winogrande_xl anli hellaswag super_glue-wic
for TASK in super_glue-wic
do
#python logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-2.yaml --templates postfix=_lr1e-6_step30_sgd_greedy_eval_fixbos-lr1e-2/${TASK} --ocl_task ${TASK} --exp_name simple
  python logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml --templates postfix=_lr1e-6_step30_greedy_eval_fixbos/${TASK} task=${TASK} --ocl_task ${TASK} --exp_name simple-0.2 --coeff 0.2 --optimizer adam
done

#for TASK in super_glue-copa hellaswag winogrande-winogrande_xl
#do
#python logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/sgd.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-2.yaml --templates postfix=_lr1e-6_step50_sgd_greedy_eval_fixbos-lr1e-2/${TASK} --ocl_task ${TASK} --exp_name simple
#done
#!/bin/bash
src_task=mmlu

for task in bbh
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/bbh_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/p3/fpd/fpd_logits_defaults.yaml configs/mmlu/fpd_bbh.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml \
configs/p3/fpd/margin_rw_multi_ts.yaml configs/p3/fpd/load_model_dir.yaml \
--templates postfix=_fpd_logits_multi_odd/step100k_margin_rw0.3_lr1e-4_sc10_tok_small/bbh task=bbh --ocl_task bbh --do_eval --eval_step 1000 \
--load_model_dir runs/instance-p3-flan-t5-large/vanilla_bg100_fpd_logits_multi/step100k_margin_rw0.3_lr1e-4_sc10_tok_small/${src_task}  \
--load_model_name best_model.pt
done

#for task in bbh
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/bbh_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_bbh.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml \
#configs/p3/fpd/prior_odd.yaml \
#configs/p3/fpd/load_model_dir.yaml \
#--templates postfix=_fpd_reps_multi_odd/step100k_margin_rw0.3_lr1e-4_sc10_tok_small/bbh task=bbh --ocl_task bbh --do_eval --eval_step 1000 \
#--load_model_dir runs/instance-p3-flan-t5-large/vanilla_bg100_fpd_rep_multi/step100k_lr2e-5_sc10_ce0.03_prior_odd_tok_small/${src_task}  \
#--load_model_name best_model.pt
#done

#for task in mmlu
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml \
#configs/p3/fpd/prior_odd.yaml \
#configs/p3/fpd/load_model_dir.yaml \
#--templates postfix=_fpd_reps_multi_odd/step100k_margin_rw0.3_lr1e-4_sc10_tok_small/mmlu task=mmlu --ocl_task mmlu --do_eval --eval_step 1000 \
#--load_model_dir runs/instance-p3-flan-t5-large/vanilla_bg100_fpd_rep_multi/step100k_lr1e-4_sc10_ce0.03_prior_odd_tok_small/${src_task}  \
#--load_model_name best_model.pt
#done

#for task in mmlu
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml \
#configs/p3/fpd/ce_weight_0.03.yaml configs/p3/fpd/prior_odd.yaml \
#--templates postfix=_fpd_rep_multi/step100k_lr1e-4_sc10_ce0.03_prior_odd_tok_small/mmlu task=mmlu --ocl_task mmlu --do_eval
#done

#for task in mmlu
#do
#python thres_predict.py --config_files configs/p3/p3_default.yaml configs/p3/bbh_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_bbh.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml \
#configs/p3/fpd/prior_odd.yaml \
#configs/p3/fpd/load_model_dir.yaml \
#--templates postfix=_fpd_reps_multi_odd/step100k_margin_rw0.3_lr1e-4_sc10_tok_small/bbh task=bbh --ocl_task bbh  \
#--load_freqs "runs/flan-t5-large-mmlu-freqs.pkl"
#done

#for task in mmlu
#do
#python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml \
#configs/p3/fpd/prior_odd.yaml \
#configs/p3/fpd/load_model_dir.yaml \
#--templates postfix=_fpd_reps_multi_odd/step100k_margin_rw0.3_lr1e-4_sc10_tok_small/mmlu task=mmlu --ocl_task mmlu --do_eval --eval_step 1000 \
#--load_model_dir runs/instance-p3-flan-t5-large/vanilla_bg100_fpd_rep_multi/step100k_lr1e-4_sc10_ce0.03_prior_odd_tok_small/${src_task}  \
#--load_model_name best_model.pt
#done

for task in bbh
do
python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/bbh_512p.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_bbh.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml \
configs/p3/fpd/load_model_dir.yaml \
--templates postfix=_fpd_reps_multi_odd/step100k_margin_rw0.3_lr1e-4_sc10_tok_small/bbh task=bbh --ocl_task bbh --do_eval --eval_step 1000 \
--load_model_dir runs/instance-p3-flan-t5-large/vanilla_bg100_fpd_rep_multi/step100k_lr2e-5_sc10_ce0.03_tok_small/${src_task}
#--load_model_name model.20000.pt
done


#for task in mmlu
#do
#python thres_predict.py --config_files configs/p3/p3_default.yaml configs/p3/flan_mmlu.yaml configs/p3/instance-bart0-base-ocl/vanilla_flan_t5_large.yaml \
#configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/steps.yaml configs/p3/instance-bart0-base-ocl/lr1e-6.yaml configs/p3/instance-bart0-base-ocl/delay_opt.yaml \
#configs/p3/fpd/fpd_defaults.yaml configs/mmlu/fpd_mmlu_text.yaml configs/p3/fpd/lr1e-4.yaml configs/p3/fpd/lr_scale10.yaml configs/mmlu/t5-small.yaml configs/p3/fpd/step100k.yaml \
#configs/p3/fpd/ce_weight_0.03.yaml configs/p3/fpd/prior_odd.yaml \
#--templates postfix=_fpd_rep_multi/step100k_lr1e-4_sc10_ce0.03_prior_odd_tok_small/mmlu task=mmlu --ocl_task mmlu -_freqs "runs/flan-t5-large-mmlu-freqs.pkl"
#done
#

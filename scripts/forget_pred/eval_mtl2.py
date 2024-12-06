from itertools import combinations
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('ks', type=int, nargs='+')
    args = parser.parse_args()
    pool = ['super_glue-cb','super_glue-copa','super_glue-rte','super_glue-wsc.fixed','super_glue-wic']
    eval_pool = ['super_glue-cb', 'super_glue-copa', 'super_glue-rte', 'super_glue-wsc.fixed', 'super_glue-wic','winogrande-winogrande_xl','anli','hellaswag']

    #for k in args.ks:
    #cmbs = [_ for _ in combinations(pool, k)]
    #print("Total {}".format(len(cmbs)))

    #for cmb in cmbs:
    tasks = '+'.join(pool)
    for eval_task in eval_pool:
        # cmd = 'python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml ' \
        #       'configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml ' \
        #       'configs/p3/instance-bart0-base-ocl/lr{LR}.yaml ' \
        #       'configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step500k.yaml ' \
        #       'configs/p3/fpd/margin_rw.yaml configs/p3/fpd/lr_scale10.yaml ' \
        #       '--templates postfix=_fpd_logits_mlp_margin_mtl/step500k_eval/{mtl_tasks} task={eval_task} --ocl_task {eval_task} --do_eval ' \
        #       '--load_model_dir /home/xsjin/cl-analysis/runs/instance-p3-bart0-large/vanilla_bg100_fpd_logits_mlp_margin_mtl/step500k/super_glue-cb+super_glue-copa+super_glue-rte+super_glue-wsc.fixed+super_glue-wic'.format(mtl_tasks=tasks, eval_task=eval_task, LR='1e-6')

        cmd = 'python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml ' \
              'configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml ' \
              'configs/p3/instance-bart0-base-ocl/lr{LR}.yaml ' \
              'configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step500k.yaml ' \
              'configs/p3/fpd/margin_rw_multi_ts.yaml configs/p3/fpd/lr_scale10.yaml ' \
              '--templates postfix=_fpd_logits_mlp_margin_mtl_multi_ts/step500k_eval/{mtl_tasks} task={eval_task} --ocl_task {eval_task} --do_eval ' \
              '--load_model_dir /home/xsjin/cl-analysis/runs/instance-p3-bart0-large/vanilla_bg100_fpd_logits_mlp_margin_mtl_multi_ts/step500k/super_glue-cb+super_glue-copa+super_glue-rte+super_glue-wsc.fixed+super_glue-wic'.format(
            mtl_tasks=tasks, eval_task=eval_task, LR='1e-6')

        print(cmd)
        os.system(cmd)

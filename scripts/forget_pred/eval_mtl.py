from itertools import combinations
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ks', type=int, nargs='+')
    args = parser.parse_args()
    pool = ['super_glue-cb','super_glue-copa','super_glue-rte','super_glue-wsc.fixed','super_glue-wic']
    eval_pool = ['super_glue-cb', 'super_glue-copa', 'super_glue-rte', 'super_glue-wsc.fixed', 'super_glue-wic','winogrande-winogrande_xl','anli','hellaswag']

    for k in args.ks:
        cmbs = [_ for _ in combinations(pool, k)]
        print("Total {}".format(len(cmbs)))

        for cmb in cmbs:
            tasks = '+'.join(cmb)
            for eval_task in eval_pool:
                # cmd = 'python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml ' \
                #           'configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml ' \
                #           'configs/p3/instance-bart0-base-ocl/lr{LR}.yaml ' \
                #           'configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/step20k.yaml ' \
                #           '--templates postfix=_fpd_paired_mean_mlp_mtl/step100k/{mtl_tasks} task={eval_task} --ocl_task {eval_task} --do_eval'.format(mtl_tasks=tasks, eval_task=eval_task, LR='1e-6')
                # print(cmb)
                # print(cmd)
                # os.system(cmd)
                #
                # cmd = 'python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml ' \
                #           'configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml ' \
                #           'configs/p3/instance-bart0-base-ocl/lr{LR}.yaml ' \
                #           'configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/step20k.yaml ' \
                #           '--templates postfix=_fpd_paired_mean_mlp_mtl_balanced/step100k/{mtl_tasks} task={eval_task} --ocl_task {eval_task} --do_eval'.format(mtl_tasks=tasks, eval_task=eval_task, LR='1e-6')
                #
                # print(cmb)
                # print(cmd)
                # os.system(cmd)

                cmd = 'python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml ' \
                          'configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml ' \
                          'configs/p3/instance-bart0-base-ocl/lr{LR}.yaml ' \
                          'configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/step20k.yaml configs/p3/fpd/prior_odd.yaml ' \
                          '--templates postfix=_fpd_paired_mean_mlp_mtl/step100k_odd/{mtl_tasks} task={eval_task} --ocl_task {eval_task} --do_eval'.format(mtl_tasks=tasks, eval_task=eval_task, LR='1e-6')

                print(cmb)
                print(cmd)
                os.system(cmd)

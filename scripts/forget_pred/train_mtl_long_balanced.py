from itertools import combinations
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('k', type=int)
    args = parser.parse_args()
    pool = ['super_glue-cb','super_glue-copa','super_glue-rte','super_glue-wsc.fixed','super_glue-wic']

    cmbs = [_ for _ in combinations(pool, args.k)]
    print("Total {}".format(len(cmbs)))

    for cmb in cmbs:
        tasks = '+'.join(cmb)

        print(cmb)

        cmd = 'python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml ' \
                  'configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml ' \
                  'configs/p3/instance-bart0-base-ocl/lr{LR}.yaml ' \
                  'configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_defaults.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/mtl.yaml configs/p3/fpd/mtl_balanced.yaml ' \
                  '--templates postfix=_fpd_paired_mean_mlp_mtl_balanced/step100k/{mtl_tasks} mtl_tasks={mtl_tasks} --ocl_task debug --do_train'.format(mtl_tasks=tasks, LR='1e-6')
        print(cmd)
        os.system(cmd)
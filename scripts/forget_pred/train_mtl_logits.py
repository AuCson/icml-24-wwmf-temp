from itertools import combinations
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ks', type=int, nargs='+')
    parser.add_argument('--one', action='store_true')
    args = parser.parse_args()
    pool = ['super_glue-cb','super_glue-copa','super_glue-rte','super_glue-wsc.fixed','super_glue-wic']

    for k in args.ks:
        cmbs = [_ for _ in combinations(pool, k)]
        print("Total {}".format(len(cmbs)))

        for cmb in cmbs:
            tasks = '+'.join(cmb)

            pt_file = os.path.join('/home/xsjin/cl-analysis/runs/instance-p3-bart0-large/vanilla_bg100_fpd_logits_mlp_margin_mtl_multi_ts/step500k',tasks,'best_model.pt')
            if os.path.isfile(pt_file):
                print('Skipping {}'.format(tasks))
                continue
            print(cmb)

            cmd = 'python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml ' \
                      'configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml ' \
                      'configs/p3/instance-bart0-base-ocl/lr{LR}.yaml ' \
                      'configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step500k.yaml ' \
                      'configs/p3/fpd/margin_rw_multi_ts.yaml configs/p3/fpd/lr_scale10.yaml configs/p3/fpd/mtl_balanced.yaml ' \
                      '--templates postfix=_fpd_logits_mlp_margin_mtl_multi_ts/step500k/{mtl_tasks} mtl_tasks={mtl_tasks} --ocl_task debug --do_train --eval_step 5000'.format(mtl_tasks=tasks, LR='1e-6')
            print(cmd)
            os.system(cmd)

            if args.once:
                print('only once')
                break
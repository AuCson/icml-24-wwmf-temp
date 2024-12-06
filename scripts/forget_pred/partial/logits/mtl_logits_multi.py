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

        pt_file = os.path.join('/home/xsjin/cl-analysis/runs/instance-p3-bart0-large/vanilla_bg100_fpd_logits_singlets_partial_mtl/step100k',tasks,'best_model.pt')
        if os.path.isfile(pt_file):
            print('Skipping {}'.format(tasks))
            continue
        print(cmb)

        PART = 'head'
        LR = 'lr1e-3'
        STEP = '30'

        cmd = 'python train_logit_predict.py --config_files configs/p3/p3_default.yaml configs/p3/instance-bart0-base-ocl/vanilla_bg100_large_eval_mode.yaml ' \
                  'configs/p3/instance-bart0-base-ocl/greedy.yaml configs/p3/instance-bart0-base-ocl/fix_label_bos.yaml configs/p3/instance-bart0-base-ocl/steps.yaml ' \
                  'configs/p3/instance-bart0-base-ocl/lr1e-6.yaml ' \
                  'configs/p3/instance-bart0-base-ocl/delay_opt.yaml configs/p3/fpd/fpd_logits_defaults.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/mtl_balanced.yaml ' \
                  'configs/p3/fpd/multi_token_kl.yaml configs/p3/fpd/step100k.yaml configs/p3/fpd/margin_rw_multi_ts.yaml configs/p3/fpd/lr_scale10.yaml configs/p3/fpd/bart_text_partial.yaml ' \
                  '--templates postfix=_fpd_logits_multits_partial_mtl/step100k/{mtl_tasks} mtl_tasks={mtl_tasks} ' \
                  'PART={part} LR={lr} STEP={step} --ocl_task debug --do_train --eval_step 5000'.format(mtl_tasks=tasks, step=STEP, part=PART, lr=LR)
        print(cmd)
        os.system(cmd)
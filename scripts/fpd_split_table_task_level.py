import argparse
import csv
import json
import numpy as np
import os
import random
import pickle
from utils.config import load_configs

def get_ocl_task_info(args):
    task_infos = []
    for task_cat, task_split in zip(args.ocl_task_cats, args.ocl_task_splits):
        if task_cat == 'mmlu':
            tasks = config.mmlu_tasks
        elif task_cat == 'bbh':
            tasks = config.bbh_tasks
        for task in tasks:
            task_infos.append({
                'cat': task_cat,
                'name': task,
                'split': task_split
            })
    return task_infos

def get_pt_task_info(args):
    task_infos = []
    if args.pt_task_cat == 'tulu':
        tasks = None
        split = None
    #for task in tasks:
    return {
        'cat': args.pt_task_cat,
        'names': tasks,
        'split': args.pt_task_split
    }

def make_split_mat_only():
    ocl_task_infos = get_ocl_task_info(args)
    raw_forget_mat = np.load(args.forget_mat)
    ocl_idxs = [_ for _ in range(raw_forget_mat.shape[0])]
    pt_task_infos = get_pt_task_info(args)

    print(len(ocl_task_infos), raw_forget_mat.shape, len(pt_task_infos))

    random.Random(SEED).shuffle(ocl_idxs)

    train_ocl_idxs, test_ocl_idxs = ocl_idxs[:int(len(ocl_idxs) * args.ratio)], ocl_idxs[int(len(ocl_idxs) * args.ratio):]

    train_mat = raw_forget_mat[train_ocl_idxs,:]
    test_mat = raw_forget_mat[test_ocl_idxs,:]

    train_ocl_task_infos = [ocl_task_infos[x] for x in train_ocl_idxs]
    test_ocl_task_infos = [ocl_task_infos[x] for x in test_ocl_idxs]

    with open(output_file,'wb') as wf:
        pickle.dump({
            'train_ocl_idxs': train_ocl_idxs,
            'test_ocl_idxs': test_ocl_idxs,
            'train_mat': train_mat,
            'test_mat': test_mat,
            'train_ocl_task_info': train_ocl_task_infos,
            'test_ocl_task_info': test_ocl_task_infos,
            'pt_task_info': pt_task_infos
        }, wf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ocl_task_cats', nargs='*')
    parser.add_argument('--ocl_task_splits', nargs='*')
    parser.add_argument('--pt_task_cat')
    parser.add_argument('--pt_task_split')
    parser.add_argument('--config_files', nargs='*')
    parser.add_argument('--forget_mat')
    parser.add_argument('--ratio', default=0.7)
    parser.add_argument('--output_file')
    args = parser.parse_args()

    SEED = 0

    output_file = args.output_file
    if os.path.exists(output_file):
        raise FileExistsError

    config = load_configs(*args.config_files)

    make_split_mat_only()




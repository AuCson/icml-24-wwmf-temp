import argparse
import csv
import json
import numpy as np
import os
import random
import pickle

def make_split_with_ds():
    with open(args.pt_ds) as f:
        reader = csv.reader(f)
        pt_rows = [_ for _ in reader]

    with open(args.ocl_ds) as f:
        reader = csv.reader(f)
        ocl_rows = [_ for _ in reader]

    raw_forget_mat = np.load(args.forget_mat)

    if args.base_correct:
        with open(args.base_correct,'rb') as f:
            base_correct = pickle.load(f)
        pt_correct_rows = [pt_rows[x] for x in base_correct]
        forget_mat = raw_forget_mat[:, base_correct]
    else:
        pt_correct_rows = pt_rows
        forget_mat = raw_forget_mat


    assert forget_mat.shape[0] == len(ocl_rows) and forget_mat.shape[1] == len(pt_correct_rows)

    ocl_idxs = [_ for _ in range(len(ocl_rows))]
    random.Random(SEED).shuffle(ocl_idxs)

    train_ocl_idxs, test_ocl_idxs = ocl_idxs[:int(len(ocl_rows) * args.ratio)], ocl_idxs[int(len(ocl_rows) * args.ratio):]

    train_ocl_rows = [ocl_rows[x] for x in train_ocl_idxs]
    test_ocl_rows = [ocl_rows[x] for x in test_ocl_idxs]

    train_mat = forget_mat[train_ocl_idxs,:]
    test_mat = forget_mat[test_ocl_idxs,:]

    with open(output_file,'wb') as wf:
        pickle.dump({
            'train_ocl_idxs': train_ocl_idxs,
            'test_ocl_idxs': test_ocl_idxs,
            'pt_correct_rows': pt_correct_rows,
            'train_ocl_rows': train_ocl_rows,
            'test_ocl_rows': test_ocl_rows,
            'train_mat': train_mat,
            'test_mat': test_mat
        }, wf)

def make_split_mat_only():
    raw_forget_mat = np.load(args.forget_mat)
    ocl_idxs = [_ for _ in range(raw_forget_mat.shape[0])]
    random.Random(SEED).shuffle(ocl_idxs)
    train_ocl_idxs, test_ocl_idxs = ocl_idxs[:int(len(ocl_idxs) * args.ratio)], ocl_idxs[
                                                                                int(len(ocl_idxs) * args.ratio):]
    train_mat = raw_forget_mat[train_ocl_idxs,:]
    test_mat = raw_forget_mat[test_ocl_idxs,:]
    with open(output_file,'wb') as wf:
        pickle.dump({
            'train_ocl_idxs': train_ocl_idxs,
            'test_ocl_idxs': test_ocl_idxs,
            'train_mat': train_mat,
            'test_mat': test_mat
        }, wf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_ds')
    parser.add_argument('--ocl_ds')
    parser.add_argument('--forget_mat')
    parser.add_argument('--base_correct')
    parser.add_argument('--ratio', default=0.7)
    parser.add_argument('--output_file')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()

    SEED = 0

    output_file = args.output_file
    if os.path.exists(output_file):
        raise FileExistsError

    if args.partial:
        make_split_mat_only()
    else:
        make_split_with_ds()




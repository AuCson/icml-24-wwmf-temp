from surprise import Dataset, SVD, BaselineOnly, NMF, SVDpp, SlopeOne, KNNBaseline, KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.reader import Reader
import random
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import ndcg_score, f1_score, mean_squared_error
import pandas as pd
import logging
import argparse
import os
import pickle
import random
from sklearn.cluster import SpectralClustering
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

clus_idxs = None

def get_cluster(arr):
    print('Getting cluster')
    arr_t = arr.transpose((1,0))
    clustering = SpectralClustering(n_clusters=2,
                                    assign_labels='discretize',
                                    random_state=0).fit(X)

def create_masked_arr(arr, idx, pt_idxs, impute, topk):
    global clus_idxs
    if impute:
        new_arr = np.zeros(arr.shape[1], dtype=np.float64)
    else:
        new_arr = np.full(arr.shape[1], -10000.0)
    if args.po == 'rand':
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(np.arange(arr.shape[1]), topk)
        new_arr[idxs] = arr[idx, idxs]
    elif args.po == 'top':
        new_arr[pt_idxs[:topk]] = arr[idx, pt_idxs[:topk]]
    elif args.po == 'clus':
        if clus_idxs is None:
            pass

    return new_arr


def create_loo_ds(train_mat, test_mat, test_idx, pt_idxs, impute, topk):
    masked_arr = create_masked_arr(test_mat, test_idx, pt_idxs, impute, topk)
    loo_mat = np.concatenate([train_mat, masked_arr.reshape(1, -1)])
    return loo_mat


def mat_to_interaction_df(mat, reader):
    data = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ##if mat[i,j] > 1e-8:
            if mat[i, j] > args.thres:
                data.append([i, j, mat[i, j]])
            # else:
            #    print(i,j,mat[i,j])
    df = pd.DataFrame(data)
    return df


def get_preds(model, ocl_idx, max_pt):
    scores = []
    for pt_idx in range(max_pt):
        scores.append(model.predict(ocl_idx, pt_idx).est)
    scores = np.array(scores)
    return scores


# main exp function
def evaluate_for_all_ocl_ds(train_mat, test_mat, topk, algo_cls, options, impute):
    all_preds = []
    reader = Reader(rating_scale=(-100., 100.))
    for test_ocl_idx in range(test_mat.shape[0]):
        print(test_ocl_idx)
        algo = algo_cls(**options)
        loo_mat = create_loo_ds(train_mat, test_mat, test_ocl_idx, pt_idxs, impute, topk)
        loo_df = mat_to_interaction_df(loo_mat, reader)
        loo_ds = Dataset.load_from_df(loo_df, reader)
        loo_trainset = loo_ds.build_full_trainset()
        algo.fit(loo_trainset)
        loo_preds = get_preds(algo, loo_trainset.n_users - 1, loo_mat.shape[1])
        all_preds.append(loo_preds)
    all_preds = np.stack(all_preds)
    return all_preds


# main exp function
def evaluate_baseline_all_ocl_ds(train_mat, test_mat, topk, algo_cls, impute):
    all_preds = []
    reader = Reader(rating_scale=(-100., 100.))
    for test_ocl_idx in range(test_mat.shape[0]):
        print(test_ocl_idx)
        algo = algo_cls()
        loo_mat = create_loo_ds(train_mat, test_mat, test_ocl_idx, pt_idxs, impute, topk)
        loo_df = mat_to_interaction_df(loo_mat, reader)
        loo_ds = Dataset.load_from_df(loo_df, reader)
        loo_trainset = loo_ds.build_full_trainset()
        algo.fit(loo_trainset)
        loo_preds = get_preds(algo, loo_trainset.n_users + 1, loo_trainset.n_items)
        all_preds.append(loo_preds)
    return all_preds


def compute_ndcg(gts, preds, k=50):
    scores = []
    for i in range(len(preds)):
        s = ndcg_score(gts[i].reshape(1, -1), preds[i].reshape(1, -1), k=k, ignore_ties=True)
        scores.append(s)
    return scores

def compute_f1(gts, preds, thres=0.):
    gts_bin = gts > args.pred_thres
    preds = np.stack(preds)

    preds_bin = preds > thres
    f1 = f1_score(gts_bin.reshape(-1), preds_bin.reshape(-1))
    return f1

def compute_rmse(gts, preds):
    preds = np.stack(preds)
    rmse = np.sqrt(mean_squared_error(gts, preds))
    return rmse

def create_split_from_mat(arr):
    rows = [_ for _ in range(arr.shape[0])]
    random.shuffle(rows)
    train_rows = arr[:int(0.75 * len(rows))]
    test_rows = arr[int(0.75 * len(rows)):]
    return train_rows, test_rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('split_file')
    parser.add_argument('--known_k', default=30, type=int)
    parser.add_argument('--po', default='rand')
    parser.add_argument('--thres', default=0., type=float)
    parser.add_argument('--pred_thres', default=0., type=float)
    parser.add_argument('--impute', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    output_dir = 'runs_rand_pts_instance_ocl/fpd_collab_filter_0615/{}'.format(args.split_file)
    os.makedirs(output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, '{}_k{}_thres{}.txt'.format(args.method, args.known_k, args.thres))))
    logger.info(repr(args.__dict__))
    with open('/home/xsjin/cl-analysis/cf_cache/{}.pkl'.format(args.split_file), 'rb') as f:
        fpd_split = pickle.load(f)
    train_mat, test_mat = fpd_split['train_mat'], fpd_split['test_mat']
    train_mat = np.nan_to_num(train_mat)
    test_mat = np.nan_to_num(test_mat)
    pt_forgets = train_mat.sum(0)
    pt_idxs = pt_forgets.argsort()
    pt_idxs = pt_idxs[::-1]

    options = {}
    if args.method == 'svd':
        algo = SVD
        options['n_epochs'] = 100
    elif args.method == 'knn':
        algo = KNNBasic
    elif args.method == 'knnb':
        algo = KNNBaseline
    elif args.method == 'baseline':
        algo = BaselineOnly
        options['bsl_options'] = {'n_epochs': 100}
    elif args.method == 'nmf':
        algo = NMF
    else:
        raise NotImplementedError
    all_preds = evaluate_for_all_ocl_ds(train_mat, test_mat, args.known_k, algo, options, impute=args.impute)


    res = {k:v for k,v in args.__dict__.items()}
    rmse = compute_rmse(test_mat, all_preds)
    #print('RMSE: {}'.format(rmse))
    #logger.info('RMSE: {}'.format(rmse))
    res['rmse_score'] = rmse
    res['f1_score'] = {}
    for thres in np.arange(-0.1,0.1,0.01):
        f1 = compute_f1(test_mat, all_preds, thres)
        #logger.info('Thres {}/F1 {}'.format(thres, f1))
        res['f1_score'][thres] = f1
        #svd_scores = compute_ndcg(test_mat, all_preds, 50)
    logger.info(json.dumps(res))

    res_dir = os.path.join(output_dir, 'preds_{}_k{}_thres{}_res'.format(args.method, args.known_k, args.thres))
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, 'preds_seed_{}.pkl'.format(args.seed)),
              'wb') as wf:
        pickle.dump(all_preds, wf)
    with open(os.path.join(res_dir, 'score_seed_{}.json'.format(args.seed)),'w') as wf:
        json.dump(res, wf)

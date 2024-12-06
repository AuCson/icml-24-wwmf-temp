import numpy as np
from tqdm import tqdm

def get_pred_thres(pred_logits_obj, pt_ds, pt_idx, ocl_error_idx, tokenizer, ts=3):
    p_logits = pred_logits_obj['pred_logits'][ocl_error_idx][pt_idx][ts]
    p_logits_idxs = pred_logits_obj['pred_logits_idxs'][ocl_error_idx][pt_idx][ts]
    labels_t = pt_ds[pt_idx]['labels'][ts]

    alt_idx, alt_score = 0,-1
    label_score = -1
    for k in range(len(p_logits)):
        if alt_score == -1 and p_logits_idxs[k] != 0 and tokenizer.convert_ids_to_tokens(int(p_logits_idxs[k])).lower() != tokenizer.convert_ids_to_tokens(labels_t).lower():
            alt_idx, alt_score = p_logits_idxs[k], p_logits[k]
        if int(p_logits_idxs[k]) == labels_t:
            label_score = p_logits[k]
        if alt_score != -1 and label_score != -1:
            break
    return alt_idx, alt_score, labels_t, label_score

def get_pred_thres_multi_ts(config, pred_logits_obj, pt_ds, pt_idx, ocl_error_idx, tokenizer):
    all_alt_idxs, all_alt_scores, all_labels, all_label_scores, all_tss = [],[],[],[], []
    for ts in range(len(pt_ds[pt_idx]['labels'])):
        label_t = pt_ds[pt_idx]['labels'][ts]
        #print(label_t)
        #print(pred_logits_obj['pred_logits'][ocl_error_idx][pt_idx])
        if label_t not in [tokenizer.pad_token_id]: # special tokens
            alt_idx, alt_score, _, label_score = get_pred_thres(pred_logits_obj, pt_ds, pt_idx, ocl_error_idx, tokenizer, ts=ts)
            all_alt_idxs.append(alt_idx)
            all_alt_scores.append(alt_score)
            all_labels.append(label_t)
            all_label_scores.append(label_score)
            all_tss.append(ts)
    return all_alt_idxs, all_alt_scores, all_labels, all_label_scores, all_tss

def get_all_alt_scores(config, pred_logits_obj, train_concat_pt_ds, tokenizer):
    # manually reduce
    #label_score_grid = np.zeros((len(ocl_error_idxs), len(train_concat_pt_ds)))
    #alt_score_grid =  np.zeros((len(ocl_error_idxs), len(train_concat_pt_ds)))
    res = {}
    ocl_error_n = len(pred_logits_obj['pred_logits'])
    for ocl_error_idx in tqdm(range(ocl_error_n), total=ocl_error_n):
        res[ocl_error_idx] = {}
        for pt_idx in range(len(train_concat_pt_ds)):
            all_alt_idxs, all_alt_scores, all_labels, all_label_scores, all_tss = get_pred_thres_multi_ts(config, pred_logits_obj, train_concat_pt_ds, pt_idx, ocl_error_idx, tokenizer)
            #label_score, alt_score = reduce_scores(all_alt_idxs, all_alt_scores, all_labels, all_label_scores, all_tss)
            #label_score_grid[ocl_error_idx, pt_idx] = label_score
            #alt_score_grid[ocl_error_idx, pt_idx] = alt_score
            #res[ocl_error_idx][pt_idx] = all_alt_idxs, all_alt_scores, all_labels, all_label_scores, all_tss
            res[ocl_error_idx][pt_idx] = {
                'all_alt_idxs': all_alt_idxs,
                'all_alt_scores': all_alt_scores,
                'all_labels': all_labels,
                'all_label_scores': all_label_scores,
                'all_tss': all_tss
            }
    return res

def get_pred_grid_em(config,all_pred_pt_logits, all_pred_pt_logits_idxs, concat_pt_ds, tokenizer):
    pred_grid = np.zeros((len(all_pred_pt_logits_idxs), len(concat_pt_ds)))
    for ocl_error_idx in range(len(all_pred_pt_logits_idxs)):
        for pt_idx in range(len(concat_pt_ds)):
            after_pred = tokenizer.decode(all_pred_pt_logits_idxs[ocl_error_idx][pt_idx][config.fpd.ts:,0],skip_special_tokens=True)
            gt = concat_pt_ds.labels[pt_idx]
            score = concat_pt_ds.compute_score_single(gt, after_pred)
            if score[0] < 1:
                pred_grid[ocl_error_idx, pt_idx] = 1
    return pred_grid


def get_pred_grid_reduce(alt_scores, train_concat_pt_ds, config, method='ts3', **kwargs):
    # manually reduce
    ocl_error_n = len(alt_scores)
    label_score_grid = np.zeros((ocl_error_n, len(train_concat_pt_ds)))
    alt_score_grid =  np.zeros((ocl_error_n, len(train_concat_pt_ds)))
    for ocl_error_idx in tqdm(range(ocl_error_n), total=ocl_error_n):
        for pt_idx in range(len(train_concat_pt_ds)):
            #all_alt_idxs, all_alt_scores, all_labels, all_label_scores, all_tss = alt_scores[ocl_error_idx][pt_idx]
            label_score, alt_score = reduce_scores(**alt_scores[ocl_error_idx][pt_idx], config=config, method=method, **kwargs)
            label_score_grid[ocl_error_idx, pt_idx] = label_score
            alt_score_grid[ocl_error_idx, pt_idx] = alt_score
    return label_score_grid, alt_score_grid

def reduce_scores(all_alt_idxs, all_alt_scores, all_labels, all_label_scores, all_tss, method, config, **kwargs):
    if method == 'ts3':
        idx = all_tss.index(config.fpd.ts)
        label_score, alt_score = all_label_scores[idx], all_alt_scores[idx]
    elif method == 'min_gap':
        gaps = [x-y for x,y in zip(all_label_scores[config.fpd.ts:], all_alt_scores[config.fpd.ts:])]
        min_idx = np.argmin(gaps) + config.fpd.ts
        label_score, alt_score = all_label_scores[min_idx], all_alt_scores[min_idx]
    elif method == 'mean':
        #gaps = [x-y for x,y in zip(all_label_scores, all_alt_scores)]
        #min_idx = np.argmin(gaps)
        #idx3 = all_tss.index(3)
        label_score, alt_score = np.mean(all_label_scores[config.fpd.ts:]),np.mean(all_alt_scores[config.fpd.ts:])
    elif method == 'mean2':
        label_score, alt_score = np.mean(all_label_scores[config.fpd.ts:-1]),np.mean(all_alt_scores[config.fpd.ts:-1])
    else:
        raise NotImplementedError
    return label_score, alt_score

def get_pred_grid(pred_logits_obj, train_concat_pt_ds, tokenizer):
    ocl_error_n = len(pred_logits_obj['pred_logits'])
    label_score_grid = np.zeros((ocl_error_n, len(train_concat_pt_ds)))
    alt_score_grid = np.zeros((ocl_error_n, len(train_concat_pt_ds)))
    for ocl_error_idx in range(ocl_error_n,):
        for pt_idx in range(len(train_concat_pt_ds)):
            alt_idx, alt_score, label_idx, label_score = get_pred_thres(pred_logits_obj, train_concat_pt_ds, pt_idx, ocl_error_idx, tokenizer)
            label_score_grid[ocl_error_idx, pt_idx] = label_score
            alt_score_grid[ocl_error_idx, pt_idx] = alt_score
    return label_score_grid, alt_score_grid
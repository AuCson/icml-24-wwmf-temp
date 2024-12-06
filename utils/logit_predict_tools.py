from utils.analysis_tools import *
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score

def load_features(exp='vanilla_bg100_lr1e-6', task='super_glue-cb', common_task='super_glue-cb', model_type='bart0'):
    with open(f'/home/xsjin/cl-analysis/runs/instance-p3-{model_type}/{exp}/{common_task}/features/pt.pkl', 'rb') as f:
        pt_feats = pickle.load(f)

    with open(f'/home/xsjin/cl-analysis/runs/instance-p3-{model_type}/{exp}/{task}/features/ocl_errors.pkl', 'rb') as f:
        ocl_error_feats = pickle.load(f)

    return pt_feats, ocl_error_feats


def predict_forgotten_examples(all_ocl_update_logits, all_pt_logits, concat_pt_ds, ocl_error_idx, ts_idx=1):
    ocl_logits_change = all_ocl_update_logits['logits_change'][ocl_error_idx]
    preds_forget = []
    for pt_idx in range(len(concat_pt_ds)):
        # print(f'Processing {pt_idx}')
        # pt_idx = 32

        pt_logit_scores, pt_logit_idxs = all_pt_logits['logits'][pt_idx]
        pt_confused_scores = pt_logit_scores[ts_idx]
        pt_confused_idxs = pt_logit_idxs[ts_idx]

        ocl_logits_change_ss = ocl_logits_change[ts_idx, pt_confused_idxs]

        # ocl_logits_change_ss[0] = ocl_logits_change_ss[0] + 1
        # if pt_confused_idxs[0] == 0:
        #    ocl_logits_change_ss[1] = ocl_logits_change_ss[1] + 1

        pred_after_pt_logits = pt_confused_scores + 0.8 * ocl_logits_change_ss

        label = all_pt_logits['labels'][pt_idx][ts_idx]

        max_idx = np.argmax(pred_after_pt_logits)

        # ignore 0
        if pt_confused_idxs[max_idx] == 0:
            pred_after_pt_logits[max_idx] = -100
            max_idx = np.argmax(pred_after_pt_logits)

            # print(pt_confused_idxs[max_idx], label)
        if pt_confused_idxs[max_idx] != label:
            preds_forget.append(pt_idx)
    return preds_forget


def predict_forgotten_examples_with_reps(all_ocl_update_logits, all_pt_logits, all_ocl_error_feats, all_pt_feats,
                                         concat_pt_ds, ocl_error_idx, ts_idx=1, thres=-1):
    ocl_logits_change = all_ocl_update_logits['logits_change'][ocl_error_idx]
    preds_forget = []
    for pt_idx in range(len(concat_pt_ds)):

        pt_logit_scores, pt_logit_idxs = all_pt_logits['logits'][pt_idx]
        pt_confused_scores = pt_logit_scores[ts_idx]
        pt_confused_idxs = pt_logit_idxs[ts_idx]

        ocl_logits_change_ss = ocl_logits_change[ts_idx, pt_confused_idxs]
        # ocl_logits_change_ss[0] = ocl_logits_change_ss[0] + 1
        # if pt_confused_idxs[0] == 0:
        #    ocl_logits_change_ss[1] = ocl_logits_change_ss[1] + 1

        ocl_x = all_ocl_error_feats['all_hiddens'][ocl_error_idx][ts_idx]
        pt_x = all_pt_feats['all_hiddens'][pt_idx][ts_idx]

        sim = np.dot(ocl_x, pt_x) / (np.dot(ocl_x, ocl_x) + 1e-10)
        # print(sim)

        pred_after_pt_logits = pt_confused_scores + sim * ocl_logits_change_ss

        label = all_pt_logits['labels'][pt_idx][ts_idx]

        max_idx = np.argmax(pred_after_pt_logits)

        # ignore 0
        if pt_confused_idxs[max_idx] == 0:
            pred_after_pt_logits[max_idx] = -100
            max_idx = np.argmax(pred_after_pt_logits)

            # print(pt_confused_idxs[max_idx], label)
        if pt_confused_idxs[max_idx] != label:
            preds_forget.append(pt_idx)
    return preds_forget


def predict_forgotten_examples_with_reps_multi_token(all_ocl_update_logits, all_pt_logits, all_ocl_error_feats,
                                                     all_pt_feats, concat_pt_ds, ocl_error_idx, ts_idx, thres=0):
    ocl_logits_change = torch.from_numpy(all_ocl_update_logits['logits_change'][ocl_error_idx])
    preds_forget = []
    start_idx, stop_idx = ts_idx, -1

    for pt_idx in range(len(concat_pt_ds)):
        forget_ts = []
        pt_logit_allt_scores, pt_logit_allt_idxs = torch.from_numpy(
            all_pt_logits['logits'][pt_idx][0]), torch.from_numpy(all_pt_logits['logits'][pt_idx][1])
        for pt_t in range(pt_logit_allt_scores.shape[0]):
            if not start_idx <= pt_t < pt_logit_allt_scores.shape[0] + stop_idx: continue
            pt_confused_scores, pt_confused_idxs = pt_logit_allt_scores[pt_t], pt_logit_allt_idxs[pt_t]  # [Vs], [Vs]

            ocl_logits_change_ss = ocl_logits_change[start_idx:stop_idx, pt_confused_idxs]  # [T2, Vs]
            # ocl_logits_change_ss = torch.gather(ocl_logits_change_ss, 1, pt_confused_idxs.unsqueeze(0))

            # print(pt_idx, ocl_logits_change_ss.shape)

            # ocl_logits_change_ss[0] = ocl_logits_change_ss[0] + 1
            # if pt_confused_idxs[0] == 0:
            #    ocl_logits_change_ss[1] = ocl_logits_change_ss[1] + 1

            ocl_x = torch.from_numpy(all_ocl_error_feats['all_hiddens'][ocl_error_idx][start_idx:stop_idx])  # [T2,H]
            pt_x = torch.from_numpy(all_pt_feats['all_hiddens'][pt_idx][pt_t])  # [H]

            sim = torch.matmul(pt_x.unsqueeze(0), ocl_x.transpose(0, 1)) / (pt_x ** 2).sum(-1)  # [1, T2]

            pred_after_pt_logits = pt_confused_scores + torch.matmul(sim, ocl_logits_change_ss).squeeze(0)  # [Vs], [1, T2] * [T2, Vs] -> [Vs]

            label = all_pt_logits['labels'][pt_idx][pt_t]

            pred_after_pt_logits = pred_after_pt_logits.detach().cpu().numpy()

            max_idx = np.argmax(pred_after_pt_logits)

            if pt_confused_idxs[max_idx] == 0:
                pred_after_pt_logits[max_idx] = -100
                max_idx = np.argmax(pred_after_pt_logits)

            # print(pt_confused_idxs[max_idx], label)
            if pt_confused_idxs[max_idx] != label:
                forget_ts.append(pt_t)
        print(pt_idx, forget_ts, ocl_x.shape, pt_x.shape)
        if len(forget_ts) > thres:
            preds_forget.append(pt_idx)
    return preds_forget


def get_gt_forget(records, all_ocl_idxs, ocl_error_idx):
    ocl_idx = all_ocl_idxs[ocl_error_idx]
    return get_new_errors(records['aff_obj']['before'], records['aff_obj'][ocl_idx])


def get_gt_label_len(all_pt_logits):
    label_lens = np.zeros(len(all_pt_logits['labels']), dtype=np.int32)
    for idx, x in enumerate(all_pt_logits['labels']):
        label_lens[idx] = len(x) - 2
    return label_lens


def evaluate_metrics(gts, preds, sz, *ignores):
    gts_arr = np.zeros(sz, dtype=np.int32)
    gts_arr[gts] = 1
    preds_arr = np.zeros(sz, dtype=np.int32)
    preds_arr[preds] = 1
    mask = np.ones(sz, dtype=bool)
    for ignl in ignores:
        for idx in ignl:
            mask[idx] = 0

    gta = gts_arr[mask]
    preda = preds_arr[mask]
    f1 = f1_score(gta, preda)
    p, r = precision_score(gta, preda), recall_score(gta, preda)
    return f1, p, r, gts_arr, preds_arr, mask


def stat_missed_logit_changes(model, optimizer, trainer, ocl_error_idx, pt_idxs, pt_topis, concat_pt_ds, ocl_error_ds,
                              collator, tokenizer):
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    optim_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state_dict().items()}

    ocl_error_example = ocl_error_ds[ocl_error_idx]
    ocl_error_batch = fix_batch(make_batch(ocl_error_example, collator), tokenizer)

    model.eval()
    trainer.nstep_model_update(model, ocl_error_batch, optimizer, n_step=config.ocl_steps, eval_mode=True)

    with torch.no_grad():
        model.eval()
        after_logits = []
        for idx, pt_idx in enumerate(pt_idxs):
            pt_batch = fix_batch(make_batch(concat_pt_ds[pt_idx], collator), tokenizer)
            topis = pt_topis[idx]  # [T,Vs]
            logits = get_logit_batch(config, model, trainer, pt_batch)  # [B,T,V]
            assert logits.size(0) == 1
            keep_mask = pt_batch['labels'][0].ne(-100)
            logits = logits[0][keep_mask]  # [Ts,V]
            print(topis.shape, logits.size())
            assert topis.shape[0] == logits.size(0)
            topi_logits = []
            for t in range(logits.size(0)):
                topi_logits.append(logits[t][topis[t]])
            topi_logits = torch.stack(topi_logits, 0).cpu().numpy()
            after_logits.append(topi_logits)

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)

    return after_logits


def update_on_example_eval_mode(config, model, trainer, batch, optimizer):
    # config.ocl_steps = 30
    # config.max_grad_norm = -1
    logger.info(f'Doing {config.ocl_steps} step of updates')
    trainer.nstep_model_update(model, batch, optimizer, n_step=config.ocl_steps, eval_mode=True)


def do_predict(ocl_error_idx, all_ocl_update_logits, all_pt_logits, all_ocl_idxs, concat_pt_ds, records, base_errors,
               other_ocl_error_idx=None, use_inner_prod=False, **kwargs):
    ts_idx = 3
    if use_inner_prod:
        all_ocl_error_feats, all_pt_feats = kwargs['all_ocl_error_feats'], kwargs['all_pt_feats']
        thres = kwargs['thres']
        preds_forget_ = predict_forgotten_examples_with_reps_multi_token(all_ocl_update_logits, all_pt_logits, all_ocl_error_feats,
                                                             all_pt_feats, concat_pt_ds, ocl_error_idx, ts_idx=ts_idx,
                                                             thres=thres)
    else:
        preds_forget_ = predict_forgotten_examples(all_ocl_update_logits, all_pt_logits, concat_pt_ds, ocl_error_idx,
                                                   ts_idx=ts_idx)
    base_error = records['aff_obj']['before']
    preds_forget = get_new_errors(base_error, preds_forget_)

    other_ocl_error_idx = ocl_error_idx + 1 if other_ocl_error_idx is None else other_ocl_error_idx
    gt_forget_other = get_gt_forget(records, all_ocl_idxs, other_ocl_error_idx)
    gt_forget = get_gt_forget(records, all_ocl_idxs, ocl_error_idx)

    label_lens = get_gt_label_len(all_pt_logits)
    lens_filter = [idx for idx, v in enumerate(label_lens) if v > 13]

    preds_forget = sorted(list(set(preds_forget).difference(base_errors)))
    gt_forget = sorted(list(set(gt_forget).difference(base_errors)))
    gt_forget_other = sorted(list(set(gt_forget_other).difference(base_errors)))

    return preds_forget, gt_forget, gt_forget_other, lens_filter
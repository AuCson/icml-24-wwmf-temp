from data_utils.p3 import P3Dataset
from data_utils.fpd_helper import FpdP3Helper, DataCollatorWithPaddingStrForFpd
from utils.config import load_configs
from transformers import AutoTokenizer
import argparse
import numpy as np
import torch
import logging
import os
import pickle

logger = logging.getLogger()

def predict_thres_based(freqs, perc):
    freq_vs = sorted([_ for _ in freqs.values()])
    thres = np.percentile(freq_vs,perc)
    preds = [k for k,v in freqs.items() if v > thres]
    #print(len(preds), thres)
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')
    parser.add_argument("--ocl_task")
    parser.add_argument("--save_freqs", default="")
    parser.add_argument("--load_freqs", default="")

    args = parser.parse_args()

    #config, base_model, tokenizer, base_trainer, collator = initialize(args.config_files, args.templates)
    config = load_configs(*args.config_files, templates=args.templates)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    fpd_collator = DataCollatorWithPaddingStrForFpd(tokenizer)
    fpd_helper = FpdP3Helper(config, tokenizer, fpd_collator, args.ocl_task)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'thres_pred_log.txt')))
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    if not args.load_freqs:
        fgt_freqs, total = fpd_helper.get_forget_freqs('train')
        if args.save_freqs:
            with open(args.save_freqs, 'wb') as f:
                freq_info = pickle.dump({'fgt_freqs': fgt_freqs, 'total': total}, f)
    else:
        with open(args.load_freqs,'rb') as f:
            freq_info = pickle.load(f)
            fgt_freqs, total = freq_info['fgt_freqs'], freq_info['total']

    fgt_label_grid, base_error_mask = fpd_helper.get_label_grid_and_base_error('train')
    dev_fgt_label_grid, dev_base_error_mask = fpd_helper.get_label_grid_and_base_error('dev')

    best_pred_grid = None
    best_f1 = -1

    for perc in np.arange(90, 100, 0.1):
        preds = predict_thres_based(fgt_freqs, perc)
        preds_grid = torch.zeros_like(dev_fgt_label_grid)
        preds_grid[:,preds] = 1
        preds_grid[:,dev_base_error_mask] = 0

        met_dict = fpd_helper.evaluate_metrics(dev_fgt_label_grid, preds_grid, base_error_mask)

        if met_dict['f1_mean'] > best_f1:
            best_f1, best_pred_grid = met_dict['f1_mean'], preds_grid

        met_dict['perc'] = perc
        logger.info(met_dict)

    fpd_helper.save_preds(best_pred_grid, base_error_mask, os.path.join(config.output_dir, 'thres_pred_fix'), 'dev')
    fpd_helper.save_preds(dev_fgt_label_grid, base_error_mask, os.path.join(config.output_dir, 'gt_fgt'), 'dev')
    with open(os.path.join(config.output_dir, 'thres_pred.npy'),'wb') as f:
        np.save(f, best_pred_grid.numpy())


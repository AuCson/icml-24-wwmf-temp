from utils.analysis_tools import initialize, get_base_optimizer, save_model_state, get_logit_batch, trim_batch
from utils.config import merge_config_into_args
import argparse
from transformers import TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from trainer.fgt_prediction_trainer import ForgettingPredictionModel
from trainer.fpd_pred_logit_reduce import get_all_alt_scores, reduce_scores, get_pred_grid_reduce, get_pred_grid_em
from data_utils.fpd_helper import FpdP3Helper, DataCollatorWithPaddingStrForFpd, handle_mtl
import logging
import os
import torch
from tqdm import tqdm
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logger = logging.getLogger('fpd_main')


class RepExtractor:
    def __init__(self, config):
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(config.fpd.model_name)
        self.lm = self.lm.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(config.fpd.model_name)

    def extract_lm_reps(self, input_ids, attention_mask, labels, decoder_attention_mask):
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                          output_hidden_states=True)
        decoder_input_len = decoder_attention_mask.sum(-1) # [B]

        last_layer_hidden = outputs.decoder_hidden_states[-1] # [B,T,H]
        raw_rep = last_layer_hidden[torch.arange(last_layer_hidden.size(0)),decoder_input_len - 1,:]
        return raw_rep

    def get_lm_sim_mat(self, ocl_reps, pt_reps):
        rep_prod_grid = torch.matmul(ocl_reps, pt_reps.transpose(0, 1)) / float(ocl_reps.size(1))
        return rep_prod_grid # [N, M]

    def clean_batch_for_rep(self, batch):
        #batch['labels'] = self.mask_pad_in_labels(batch['labels'])
        batch['input_ids'], batch['attention_mask'] = trim_batch(batch['input_ids'], self.tokenizer.pad_token_id, batch['attention_mask'])
        batch['labels'], batch['decoder_attention_mask'] = trim_batch(batch['labels'], self.tokenizer.pad_token_id, batch['_decoder_attention_mask'])
        batch['labels'] = self.mask_pad_in_labels(batch['labels'])
        batch = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']}

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda()

        return batch

    def mask_pad_in_labels(self, labels):
        ret = labels.masked_fill(labels==self.tokenizer.pad_token_id, -100)
        return ret

class CountVecExtractor:
    def __init__(self, config):
        self.vectorizer = TfidfVectorizer()

    def fit(self, pt_texts, ocl_texts):
        all_texts = pt_texts + ocl_texts
        all_vecs = self.vectorizer.fit_transform(all_texts)
        pt_vecs = all_vecs[:len(pt_texts)]
        ocl_vecs = all_vecs[len(pt_texts):]

        rep_mat = ocl_vecs * pt_vecs.transpose((1,0))
        rep_mat = rep_mat.toarray()
        return rep_mat

def train_fpd_model(config, fpd_helper):
    fpd_train_step = config.fpd.train_step
    bs = config.fpd.train_batch_size

    extractor = RepExtractor(config)

    all_pt_reps, all_ocl_reps = [], []

    pt_loader, ocl_loader = fpd_helper.get_pt_dataloader('train', config.fpd.eval_batch_size), \
                            fpd_helper.get_ocl_dataloader('train', config.fpd.eval_batch_size)

    with torch.no_grad():
        print('Getting PT example reps')
        for _, pt_batch in tqdm(enumerate(pt_loader), total=len(pt_loader)):
            pt_batch = extractor.clean_batch_for_rep(pt_batch)
            reps = extractor.extract_lm_reps(pt_batch['input_ids'], pt_batch['attention_mask'], pt_batch['labels'],
                                      pt_batch['decoder_attention_mask'])
            reps = reps.detach()
            all_pt_reps.append(reps)
        all_pt_reps = torch.cat(all_pt_reps, 0) # [N1,H]
        print('Getting OCL example reps')
        for _, ocl_batch in tqdm(enumerate(ocl_loader), total=len(ocl_loader)):
            ocl_batch = extractor.clean_batch_for_rep(ocl_batch)
            reps = extractor.extract_lm_reps(ocl_batch['input_ids'], ocl_batch['attention_mask'], ocl_batch['labels'],
                                      ocl_batch['decoder_attention_mask'])
            reps = reps.detach()
            all_ocl_reps.append(reps)
        all_ocl_reps = torch.cat(all_ocl_reps, 0) # [N2, H]

        rep_prod_grid = extractor.get_lm_sim_mat(all_ocl_reps, all_pt_reps).cpu().detach().numpy()
        fgt_label_grid, base_error_mask = fpd_helper.get_label_grid_and_base_error('train')

    rep_prod_masked = rep_prod_grid[:,~base_error_mask]
    fgt_label_masked = fgt_label_grid[:,~base_error_mask]

    rep_prod_flat = rep_prod_masked.reshape(-1)
    fgt_label_flat = fgt_label_masked.reshape(-1)

    model = LogisticRegression(class_weight='balanced')
    model.fit(rep_prod_flat.reshape(-1,1), fgt_label_flat)
    pred = model.predict(rep_prod_flat.reshape(-1,1))
    f1 = f1_score(fgt_label_flat, pred)
    logger.info('F1 score: {}'.format(f1))

    return model


def train_cnt_model(config, fpd_helper):
    fpd_train_step = config.fpd.train_step
    bs = config.fpd.train_batch_size

    all_pt_texts, all_ocl_texts = [], []

    pt_loader, ocl_loader = fpd_helper.get_pt_dataloader('train', config.fpd.eval_batch_size), \
                            fpd_helper.get_ocl_dataloader('train', config.fpd.eval_batch_size)

    pt_texts = [pt_loader.dataset[x]['original_input'] + ' ' + pt_loader.dataset[x]['original_answers'] for x in range(len(pt_loader.dataset))]
    ocl_texts = [ocl_loader.dataset[x]['original_input'] + ' ' + ocl_loader.dataset[x]['original_answers'] for x in range(len(ocl_loader.dataset))]

    extractor = CountVecExtractor(config)

    rep_prod_grid = extractor.fit(pt_texts, ocl_texts)
    fgt_label_grid, base_error_mask = fpd_helper.get_label_grid_and_base_error('train')

    rep_prod_masked = rep_prod_grid[:, ~base_error_mask]
    fgt_label_masked = fgt_label_grid[:, ~base_error_mask]

    rep_prod_flat = rep_prod_masked.reshape(-1)
    fgt_label_flat = fgt_label_masked.reshape(-1)

    model = LogisticRegression()
    model.fit(rep_prod_flat.reshape(-1, 1), fgt_label_flat)
    pred = model.predict(rep_prod_flat.reshape(-1, 1))
    pred_prob = model.predict_proba(rep_prod_flat.reshape(-1, 1))
    f1 = f1_score(fgt_label_flat, pred)
    logger.info('F1 score: {}'.format(f1))

    return model

def infer_fpd_model(config, fpd_model, fpd_helper: FpdP3Helper, split, save_path=None, try_thres=False):
    print('Starting inference')
    is_training = fpd_model.training
    fpd_model.eval()
    # get reps of all pt and ocl_examples
    pt_loader, ocl_loader = fpd_helper.get_pt_dataloader(split, config.fpd.eval_batch_size), \
                            fpd_helper.get_ocl_dataloader(split, config.fpd.eval_batch_size)
    fgt_label_grid, base_error_mask = fpd_helper.get_label_grid_and_base_error(split)

    all_priors = fpd_helper.get_all_priors(split)
    if all_priors is not None: all_priors = all_priors.cuda()

    # all reps
    all_pt_reps = []
    all_ocl_reps = []

    with torch.no_grad():
        print('Getting PT example reps')
        for _, pt_batch in tqdm(enumerate(pt_loader), total=len(pt_loader)):
            pt_batch = fpd_model.clean_batch_for_rep(pt_batch)
            reps = fpd_model.get_reps(pt_batch['input_ids'], pt_batch['attention_mask'], pt_batch['labels'],
                                      pt_batch['decoder_attention_mask'])
            reps = reps.detach()
            all_pt_reps.append(reps)
        all_pt_reps = torch.cat(all_pt_reps, 0) # [N1,H]
        print('Getting OCL example reps')
        for _, ocl_batch in tqdm(enumerate(ocl_loader), total=len(ocl_loader)):
            ocl_batch = fpd_model.clean_batch_for_rep(ocl_batch)
            reps = fpd_model.get_reps(ocl_batch['input_ids'], ocl_batch['attention_mask'], ocl_batch['labels'],
                                      ocl_batch['decoder_attention_mask'])
            reps = reps.detach()
            all_ocl_reps.append(reps)
        all_ocl_reps = torch.cat(all_ocl_reps, 0) # [N2, H]

        if try_thres:
            for thres in np.arange(0.4,1,0.005):
                prob_grid, preds_grid = fpd_model.pred_forget_with_reps(all_ocl_reps, all_pt_reps,
                                                                        all_priors=all_priors, thres=thres)
                met_dict = fpd_helper.evaluate_metrics(fgt_label_grid, preds_grid, base_error_mask)
                logger.info('Thres {} Metrics over {}: {}'.format(thres, split, met_dict))

        prob_grid, preds_grid = fpd_model.pred_forget_with_reps(all_ocl_reps, all_pt_reps,
                                                                all_priors=all_priors)
        met_dict = fpd_helper.evaluate_metrics(fgt_label_grid, preds_grid, base_error_mask)
    logger.info('Metrics over {}: {}'.format(split, met_dict))
    fpd_model.train(is_training)
    if save_path:
        print('Save path is', save_path)
        fpd_helper.save_preds(preds_grid, base_error_mask, save_path, split)
    return met_dict, preds_grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')
    parser.add_argument("--ocl_task")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--load_model_dir")
    parser.add_argument("--load_model_name", default='best_model.pt')
    parser.add_argument("--skip_first_eval", action='store_true')
    parser.add_argument("--return_pred_logits", action='store_true')
    parser.add_argument("--try_thres", action='store_true')

    parser.add_argument("--rep_type", default='rep')

    args = parser.parse_args()

    config, base_model, tokenizer, base_trainer, collator = initialize(args.config_files, args.templates)
    base_model = base_model.cuda()

    # mtl or not
    if config.fpd.mtl:
        mtl_tasks_str = config.fpd.mtl_tasks
        mtl_tasks = mtl_tasks_str.split('+')
        config = handle_mtl(config, mtl_tasks) # output dir will be updated

    training_args = TrainingArguments(output_dir=config.output_dir)
    merge_config_into_args(config, training_args)

    base_optimizer = get_base_optimizer(config, base_model, training_args=training_args)


    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    config.save(config.output_dir, 'config.json')

    fpd_collator = DataCollatorWithPaddingStrForFpd(tokenizer)

    fpd_helper = FpdP3Helper(config, tokenizer, fpd_collator, args.ocl_task)

    #if args.do_train:
    if args.rep_type == 'rep':
        fpd_model = train_fpd_model(config, fpd_helper)
    else:
        fpd_model = train_cnt_model(config, fpd_helper)

    #if args.do_eval:

    #    met_dict, preds_grid = infer_fpd_model(config, fpd_model, fpd_helper, split='dev', save_path=os.path.join(config.output_dir,f'fpd_dev/{args.ocl_task}'), try_thres=args.try_thres)

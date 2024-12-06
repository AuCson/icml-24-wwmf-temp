from torch.utils.data import Dataset, Subset, ConcatDataset
import torch
from .p3 import load_bg_train_ds, P3ConcatDataset, load_ocl_ds_splits, P3Dataset, save_dataset
from .lm import SFTDataset
import pickle
import os
import random as random_
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from transformers import DataCollatorWithPadding
import time
import csv
import copy
from collections import defaultdict

import logging
logger = logging.getLogger('fpd_helper')

MAX_N_PER_TASK = 10000

random = random_.Random(0)

def handle_mtl(config, mtl_tasks):
    config.output_dir = config.output_dir.format(task='+'.join(mtl_tasks))
    os.makedirs(config.output_dir, exist_ok=True)

    config = create_tmp_datasets(config, mtl_tasks, 'train')
    config = create_tmp_datasets(config, mtl_tasks, 'dev')

    config.fpd.mtl_tasks = mtl_tasks

    config.fpd.train_concat_pt_ds_dir = config.fpd.train_concat_pt_ds_dir.format(task=mtl_tasks[0])
    config.fpd.dev_concat_pt_ds_dir = config.fpd.dev_concat_pt_ds_dir.format(task=mtl_tasks[0])

    config.fpd.train_pt_logits_file = config.fpd.train_pt_logits_file.format(task=mtl_tasks[0])
    config.fpd.dev_pt_logits_file = config.fpd.dev_pt_logits_file.format(task=mtl_tasks[0])

    return config

def make_batch(ocl_example, collator):
    batch = collator([ocl_example])
    return batch

def create_tmp_datasets(config, all_ocl_tasks, split):
    merged_aff_log = {}
    merged_ocl_log = {}
    merged_ocl_update_logits = {}
    merged_ocl_error_ds = []

    for task_id, task in enumerate(all_ocl_tasks):
        with open(getattr(config.fpd,f'{split}_aff_log_path').format(task=task),'rb') as f:
            aff_log = pickle.load(f)

            merged_aff_log['before'] = aff_log['before']
            for k,v in aff_log.items():
                if k != 'before':
                    merged_aff_log[MAX_N_PER_TASK * task_id + k] = v

        with open(getattr(config.fpd,f'{split}_ocl_log_path').format(task=task),'rb') as f:
            ocl_log = pickle.load(f)

            assert max(ocl_log.keys()) < MAX_N_PER_TASK
            for k,v in ocl_log.items():
                merged_ocl_log[MAX_N_PER_TASK * task_id + k] = v

        if config.fpd.require_logits:
            with open(getattr(config.fpd,f'{split}_ocl_update_logits_file').format(task=task),'rb') as f:
                ocl_update_logits = pickle.load(f)
                for key, data in ocl_update_logits.items():
                    if key not in merged_ocl_update_logits:
                        merged_ocl_update_logits[key] = []
                    if  len(ocl_log) < len(data):
                        logger.warning(f'Warning: fewer records in {split} for ocl_update_records')
                    merged_ocl_update_logits[key].extend(data[:len(ocl_log)]) # ocl_update_logits may contain extra info

        ocl_error_ds_path = os.path.join(getattr(config.fpd,f'{split}_ocl_error_ds_dir').format(task=task), 'ocl_error_ds.csv')
        with open(ocl_error_ds_path) as f:
            reader = csv.reader(f)
            rows = [_ for _ in reader]
            merged_ocl_error_ds.extend(rows)

    # save
    merged_aff_log_save_path = os.path.join(config.output_dir, f'{split}_merged_aff_log.pkl').format(task='+'.join(all_ocl_tasks))
    merged_ocl_log_save_path = os.path.join(config.output_dir, f'{split}_merged_ocl_log.pkl').format(task='+'.join(all_ocl_tasks))
    merged_ocl_error_ds_save_dir = os.path.join(config.output_dir, f'{split}_merged_ds').format(task='+'.join(all_ocl_tasks))
    merged_ocl_update_logits_path = os.path.join(config.output_dir, f'{split}_merged_ocl_logits_change.pkl').format(task='+'.join(all_ocl_tasks))

    with open(merged_aff_log_save_path,'wb') as wf:
        pickle.dump(merged_aff_log, wf)

    with open(merged_ocl_log_save_path,'wb') as wf:
        pickle.dump(merged_ocl_log, wf)

    if config.fpd.require_logits:
        with open(merged_ocl_update_logits_path,'wb') as wf:
            pickle.dump(merged_ocl_update_logits,wf)

    os.makedirs(merged_ocl_error_ds_save_dir, exist_ok=True)
    with open(os.path.join(merged_ocl_error_ds_save_dir,'ocl_error_ds.csv'),'w') as wf:
        writer = csv.writer(wf)
        writer.writerows(merged_ocl_error_ds)

    replaced_config = copy.deepcopy(config)
    setattr(replaced_config.fpd, f'{split}_aff_log_path', merged_aff_log_save_path)
    setattr(replaced_config.fpd, f'{split}_ocl_log_path', merged_ocl_log_save_path)
    setattr(replaced_config.fpd, f'{split}_ocl_update_logits_file', merged_ocl_update_logits_path)
    setattr(replaced_config.fpd, f'{split}_ocl_error_ds_dir', merged_ocl_error_ds_save_dir)

    return replaced_config


class FpdP3Helper:
    def __init__(self, config, tokenizer, data_collator, ocl_task):
        self.config = config
        self.tokenizer = tokenizer
        self.ocl_task = ocl_task
        self.collator = data_collator

        self._global_sample_state = 0

        #self.train_pt_ds = self.prepare_concat_pt_ds(config.fpd.pt_train_ds_size, config.fpd.pt_train_offset)
        #self.dev_pt_ds = self.prepare_concat_pt_ds(config.fpd.pt_dev_ds_size, config.fpd.pt_dev_offset)
        #self.test_pt_ds = self.prepare_concat_pt_ds(config.fpd.pt_test_ds_size, config.fpd.pt_test_offset)

        if self.config.fpd.use_fpd_split_file:
            print('Load fpd pre-splitted files')
            self.train_ocl_error_ds, self.dev_ocl_error_ds, self.train_pt_ds, self.train_gt_forgets, self.dev_gt_forgets, self.train_base_errors = self.prepare_from_split()
            self.test_ocl_error_ds = self.dev_ocl_error_ds
            self.test_gt_forgets = self.dev_gt_forgets
            self.dev_pt_ds = self.test_pt_ds = self.train_pt_ds
            self.dev_base_errors = self.test_base_errors = self.train_base_errors
        elif self.config.fpd.use_fpd_split_file_task_level:
            print('Load fpd pre-splitted files - task level')
            self.train_ocl_dss, self.test_ocl_dss, self.pt_ds, self.train_mat, self.test_mat = self.prepare_from_split_sep_tasks()
            self.dev_ocl_dss, self.dev_mat = self.test_ocl_dss, self.test_mat
            self.train_pt_ds = self.dev_pt_ds = self.test_pt_ds = self.pt_ds
            self.train_ocl_error_ds = ConcatDataset(self.train_ocl_dss)
            self.test_ocl_error_ds = ConcatDataset(self.test_ocl_dss)
            self.dev_ocl_error_ds = ConcatDataset(self.dev_ocl_dss)
        else:
            self.train_pt_ds, self.dev_pt_ds, self.test_pt_ds = self.prepare_concat_pt_ds()
            self.train_ocl_error_ds, self.dev_ocl_error_ds, self.test_ocl_error_ds = self.prepare_ocl_error_ds()

            self.train_gt_forgets, self.train_base_errors = self.prepare_gt_forgets(config.fpd.train_aff_log_path, config.fpd.train_ocl_log_path)
            self.dev_gt_forgets, self.dev_base_errors = self.prepare_gt_forgets(config.fpd.dev_aff_log_path, config.fpd.dev_ocl_log_path)
            self.test_gt_forgets, self.test_base_errors = self.dev_gt_forgets, self.dev_base_errors

        self.train_pt_update_logits, self.dev_pt_update_logits = None, None

        if config.fpd.require_logits:
            self.train_pt_logits, self.train_ocl_logits,  = self.load_pt_logits(config.fpd.train_pt_logits_file, config.fpd.train_ocl_update_logits_file)
            self.dev_pt_logits, self.dev_ocl_logits = self.load_pt_logits(config.fpd.dev_pt_logits_file, config.fpd.dev_ocl_update_logits_file)
            if config.fpd.logit_loss_type == 'kl':
                if config.fpd.train_pt_update_logits_file:
                    self.train_pt_update_logits = self.load_pt_update_logits(config.fpd.train_pt_update_logits_file)
                if config.fpd.dev_pt_update_logits_file:
                    self.dev_pt_update_logits = self.load_pt_update_logits(config.fpd.dev_pt_update_logits_file)

            self.test_pt_logits, self.test_ocl_logits = self.dev_ocl_logits, self.dev_pt_logits
        else:
            self.train_ocl_logits, self.train_pt_logits, self.dev_ocl_logits, self.dev_pt_logits, \
                self.test_ocl_logits, self.test_pt_logits = None, None, None, None, None, None

        self.train_priors, self.dev_priors, self.test_priors = self.prepare_priors()

    def load_pt_logits(self, pt_logits_file, ocl_update_logits_file):
        with open(pt_logits_file,'rb') as f:
            pt_logits = pickle.load(f)
        with open(ocl_update_logits_file, 'rb') as f:
            ocl_update_logits = pickle.load(f)
        return pt_logits, ocl_update_logits

    def load_pt_update_logits(self, pt_update_logits_file):
        with open(pt_update_logits_file,'rb') as f:
            pt_update_logits = pickle.load(f)
        return pt_update_logits

    def prepare_concat_pt_ds(self):
        train_concat_pt_ds_path = os.path.join(self.config.fpd.train_concat_pt_ds_dir, 'concat_pt_ds.csv')
        train_concat_pt_ds = P3Dataset.from_csv(train_concat_pt_ds_path, self.config, self.tokenizer)

        dev_concat_pt_ds_path = os.path.join(self.config.fpd.dev_concat_pt_ds_dir, 'concat_pt_ds.csv')
        dev_concat_pt_ds = P3Dataset.from_csv(dev_concat_pt_ds_path, self.config, self.tokenizer)

        return train_concat_pt_ds, dev_concat_pt_ds, dev_concat_pt_ds

    def prepare_ocl_error_ds(self):
        train_ocl_error_ds_path = os.path.join(self.config.fpd.train_ocl_error_ds_dir, 'ocl_error_ds.csv')
        train_ocl_error_ds = P3Dataset.from_csv(train_ocl_error_ds_path, self.config, self.tokenizer)

        dev_ocl_error_ds_path = os.path.join(self.config.fpd.dev_ocl_error_ds_dir, 'ocl_error_ds.csv')
        dev_ocl_error_ds = P3Dataset.from_csv(dev_ocl_error_ds_path, self.config, self.tokenizer)

        return train_ocl_error_ds, dev_ocl_error_ds, dev_ocl_error_ds

    def prepare_from_split(self):
        with open(self.config.fpd.fpd_split_file,'rb') as f:
            fpd_split = pickle.load(f)
        train_ocl_ds, test_ocl_ds = P3Dataset.from_rows(fpd_split['train_ocl_rows'],self.config, self.tokenizer), \
                                P3Dataset.from_rows(fpd_split['test_ocl_rows'],self.config,self.tokenizer)
        pt_ds = P3Dataset.from_rows(fpd_split['pt_correct_rows'], self.config, self.tokenizer)
        train_mat, test_mat = fpd_split['train_mat'], fpd_split['test_mat']
        train_gt_forgets = self.mat_to_bin_fgt_list(train_mat, thres=0.5)
        test_gt_forgets = self.mat_to_bin_fgt_list(test_mat, thres=0.5)
        base_errors = []

        return train_ocl_ds, test_ocl_ds, pt_ds, train_gt_forgets, test_gt_forgets, base_errors

    def prepare_from_split_sep_tasks(self):
        def get_sft_dss(task_infos):
            dss = []
            for task_info in task_infos:
                task_cat, task_name, task_split = task_info['cat'], task_info['name'], task_info['split']
                ds = SFTDataset.from_auto(task_cat, tasks=[task_name], split=task_split,
                                          config=self.config,
                                          tokenizer=self.tokenizer)
                dss.append(ds)
            return dss

        with open(self.config.fpd.fpd_split_file, 'rb') as f:
            fpd_split = pickle.load(f)

        train_ocl_dss = get_sft_dss(fpd_split['train_ocl_task_info'])
        test_ocl_dss = get_sft_dss(fpd_split['test_ocl_task_info'])
        pt_ds_full = SFTDataset.from_auto(fpd_split['pt_task_info']['cat'], tasks=fpd_split['pt_task_info']['names'], split=fpd_split['pt_task_info']['split'],
                                     config=self.config, tokenizer=self.tokenizer)
        if 'ss_idxs' in fpd_split['pt_task_info']:
            pt_ds_ss_idxs = fpd_split['pt_task_info']['ss_idxs']
            pt_ds = Subset(pt_ds_full, pt_ds_ss_idxs)
        else:
            pt_ds = pt_ds_full

        train_mat, test_mat = fpd_split['train_mat'], fpd_split['test_mat']
        train_mat = np.nan_to_num(train_mat)
        test_mat = np.nan_to_num(test_mat)
        return train_ocl_dss, test_ocl_dss, pt_ds, train_mat, test_mat

    def mat_to_bin_fgt_list(self, mat, thres=1e-10):
        forgets = {}
        arr = np.arange(mat.shape[1])
        for idx in range(mat.shape[0]):
            forgets[idx] = arr[mat[idx] > thres].tolist()
        return forgets

    def prepare_gt_forgets(self, aff_log_path, ocl_log_path):
        gt_forgets = {}
        with open(aff_log_path,'rb') as f:
            aff_log = pickle.load(f)
        with open(ocl_log_path,'rb') as f:
            ocl_log = pickle.load(f)
        base_errors = aff_log['before']
        ocl_idxs = sorted([x for x in ocl_log.keys()])

        for ocl_error_idx, ocl_idx in enumerate(ocl_idxs):
            gt_forgets[ocl_error_idx] = sorted([x for x in aff_log[ocl_idx] if x not in base_errors])
        return gt_forgets, base_errors

    def prepare_priors(self):
        res = []
        for split in ['train','dev','test']:
            fixed_priors = None
            if self.config.fpd.prior == 'odd':
                fixed_priors = self.get_fixed_bias_terms(split)
            res.append(fixed_priors)
        return res

    def get_pt_dataloader(self, split, batch_size):
        ds = getattr(self, f'{split}_pt_ds')
        loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=self.collator)
        return loader

    def get_pt_ds(self, split):
        ds = getattr(self, f'{split}_pt_ds')
        return ds

    def get_ocl_dataloader(self, split, batch_size):
        ds = getattr(self, f'{split}_ocl_error_ds')
        loader = DataLoader(ds, batch_size, shuffle=False, collate_fn=self.collator)
        return loader

    def get_label_grid_and_base_error(self, split):
        ocl_error_ds = getattr(self, f'{split}_ocl_error_ds')
        pt_ds = getattr(self, f'{split}_pt_ds')
        label_grid = torch.zeros(len(ocl_error_ds), len(pt_ds))
        gt_forgets = getattr(self, f'{split}_gt_forgets')
        base_errors = getattr(self,f'{split}_base_errors')
        base_error_mask = torch.zeros(len(pt_ds)).bool()
        base_error_mask[base_errors] = True
        for ocl_error_idx in range(len(ocl_error_ds)):
            label_grid[ocl_error_idx, gt_forgets[ocl_error_idx]] = 1
        return label_grid, base_error_mask

    def get_all_pt_logits(self, split):
        return getattr(self, f'{split}_pt_logits')

    def get_all_ocl_logits_change(self, split):
        ocl_logits = getattr(self, f'{split}_ocl_logits')
        return ocl_logits['logits_change']

    def get_all_gt_forgets(self, split):
        return getattr(self, f'{split}_gt_forgets')

    def sample_episode_batch_paired(self, split, bs, extra_infos=None):
        if extra_infos is None: extra_infos = []
        pt_ds, ocl_error_ds, gt_forgets, base_errors = getattr(self, f'{split}_pt_ds'), getattr(self, f'{split}_ocl_error_ds'), \
                                          getattr(self, f'{split}_gt_forgets'), getattr(self, f'{split}_base_errors')
        ocl_logits, pt_logits = getattr(self,f'{split}_ocl_logits'), getattr(self,f'{split}_pt_logits')
        pt_update_logits = getattr(self, f'{split}_pt_update_logits')
        priors = getattr(self,f'{split}_priors')

        examples = []
        mtl_balanced = self.config.fpd.mtl_balanced
        is_mtl = self.config.fpd.mtl

        for b in range(bs):
            if is_mtl and mtl_balanced:
                tasks = self.config.fpd.mtl_tasks
                task_id = self._global_sample_state % len(tasks)
                self._global_sample_state += 1
                valid_idxs = [x for x in range(len(ocl_error_ds)) if ocl_error_ds[x]['task_name'] == tasks[task_id]]
                ocl_error_idx = random.choice(valid_idxs)
            else:
                ocl_error_idx = random.choice(range(len(ocl_error_ds)))

            ocl_example = ocl_error_ds[ocl_error_idx]

            fgt = gt_forgets[ocl_error_idx]
            non_fgt = [x for x in range(len(pt_ds)) if x not in fgt and x not in base_errors]

            fgt_pt_idx, non_fgt_pt_idx = random.choice(fgt), random.choice(non_fgt)
            fgt_pt_example, non_fgt_pt_example = pt_ds[fgt_pt_idx], pt_ds[non_fgt_pt_idx]

            pos_example = {
                'input_ids': fgt_pt_example['input_ids'], # required by tokenizer
                'input_ids_pt': fgt_pt_example['input_ids'],
                'attention_mask_pt': fgt_pt_example['attention_mask'],
                'labels_pt': fgt_pt_example['labels'],
                'decoder_attention_mask_pt': fgt_pt_example['_decoder_attention_mask'],
                'input_ids_ocl': ocl_example['input_ids'],
                'attention_mask_ocl': ocl_example['attention_mask'],
                'labels_ocl': ocl_example['labels'],
                'decoder_attention_mask_ocl': ocl_example['_decoder_attention_mask'],
                'forget_label': 1,
                'pt_idx': fgt_pt_idx,
            }

            if priors is not None:
                pos_example['priors'] = priors[fgt_pt_idx]

            neg_example = {
                'input_ids': fgt_pt_example['input_ids'],  # required by tokenizer
                'input_ids_pt': non_fgt_pt_example['input_ids'],
                'attention_mask_pt': non_fgt_pt_example['attention_mask'],
                'labels_pt': non_fgt_pt_example['labels'],
                'decoder_attention_mask_pt': non_fgt_pt_example['_decoder_attention_mask'],
                'input_ids_ocl': ocl_example['input_ids'],
                'attention_mask_ocl': ocl_example['attention_mask'],
                'labels_ocl': ocl_example['labels'],
                'decoder_attention_mask_ocl': ocl_example['_decoder_attention_mask'],
                'forget_label': 0,
                'pt_idx': non_fgt_pt_idx
            }

            if priors is not None:
                neg_example['priors'] = priors[non_fgt_pt_idx]

            if 'logits_change' in extra_infos:
                pos_pt_logit_scores, pos_pt_logit_idxs = pt_logits['logits'][fgt_pt_idx]
                neg_pt_logit_scores, neg_pt_logit_idxs = pt_logits['logits'][non_fgt_pt_idx]
                ocl_logits_change = ocl_logits['logits_change'][ocl_error_idx]

                pos_pt_logit_scores, pos_pt_logit_idxs = torch.from_numpy(pos_pt_logit_scores), torch.from_numpy(pos_pt_logit_idxs)
                neg_pt_logit_scores, neg_pt_logit_idxs = torch.from_numpy(neg_pt_logit_scores), torch.from_numpy(neg_pt_logit_idxs)
                ocl_logits_change = torch.from_numpy(ocl_logits_change)

                pos_example['pt_logits_ss'], pos_example['pt_logits_idxs'] = pos_pt_logit_scores, pos_pt_logit_idxs
                neg_example['pt_logits_ss'], neg_example['pt_logits_idxs'] = neg_pt_logit_scores, neg_pt_logit_idxs
                pos_example['ocl_update_logits'] = neg_example['ocl_update_logits'] = ocl_logits_change

            if 'pt_logits_change' in extra_infos:
                pos_logits_after_ss = pt_update_logits[ocl_error_idx]['logits'][fgt_pt_idx]
                neg_logits_after_ss = pt_update_logits[ocl_error_idx]['logits'][non_fgt_pt_idx]

                #pos_logits_after_idxs = pt_update_logits[ocl_error_idx]['labels'][fgt_pt_idx]
                #neg_logits_after_idxs = pt_update_logits[ocl_error_idx]['labels'][non_fgt_pt_idx]

                pos_example['pt_logits_after_ss'] = torch.from_numpy(pos_logits_after_ss[0])
                pos_example['pt_logits_after_idxs'] = torch.from_numpy(pos_logits_after_ss[1])

                neg_example['pt_logits_after_ss'] = torch.from_numpy(neg_logits_after_ss[0])
                neg_example['pt_logits_after_idxs'] = torch.from_numpy(neg_logits_after_ss[1])

            examples.append(pos_example)
            examples.append(neg_example)

        batch = self.collator(examples)
        #print(batch)
        return batch

    def sample_episode_random_ocl(self, split, bs):
        pt_ds, ocl_error_ds, gt_forgets, base_errors = getattr(self, f'{split}_pt_ds'), getattr(self,
                                                                                                f'{split}_ocl_error_ds'), \
                                                       getattr(self, f'{split}_gt_forgets'), getattr(self,
                                                                                                     f'{split}_base_errors')
        # ocl_logits, pt_logits = getattr(self, f'{split}_ocl_logits'), getattr(self, f'{split}_pt_logits')
        # pt_update_logits = getattr(self, f'{split}_pt_update_logits')
        # priors = getattr(self, f'{split}_priors')

        ocl_error_idx = random.choice(range(len(ocl_error_ds)))
        ocl_example = ocl_error_ds[ocl_error_idx]
        ocl_batch = make_batch(ocl_example, self.collator)

        fgt = [x for x in gt_forgets[ocl_error_idx] if x not in base_errors]
        non_fgt = [x for x in range(len(pt_ds)) if x not in fgt and x not in base_errors]

        if not fgt:
            pos_idxs = []
        elif len(fgt) > bs:
            pos_idxs = random.sample(fgt, bs)
        else:
            pos_idxs = fgt
        pos_examples = [pt_ds[idx] for idx in pos_idxs]

        if not non_fgt:
            neg_idxs = []
        elif len(non_fgt) > bs:
            neg_idxs = random.sample(non_fgt, bs)
        else:
            neg_idxs = non_fgt
        neg_examples = [pt_ds[idx] for idx in neg_idxs]

        pt_examples = pos_examples + neg_examples
        pt_batch = self.collator(pt_examples)
        return ocl_batch, pt_batch

    def sample_episode_task_level_balanced(self, split, bs):
        ocl_dss, pt_ds = getattr(self, '{}_ocl_dss'.format(split)), self.pt_ds
        fgt_mat = getattr(self, '{}_mat'.format(split))
        ocl_ds_num, pt_ex_num = len(ocl_dss), len(pt_ds)

        examples = []
        for b in range(bs):
            ocl_ds_idx = random.choice(range(ocl_ds_num))
            ocl_ds = ocl_dss[ocl_ds_idx]
            ocl_idx = random.choice(range(len(ocl_ds)))
            ocl_example = ocl_ds[ocl_idx]

            forgotten_idx = np.arange(len(pt_ds))[fgt_mat[ocl_ds_idx] > 0]
            non_forgotten_idx = np.arange(len(pt_ds))[fgt_mat[ocl_ds_idx] <= 0]

            if len(forgotten_idx) > 0:
                pos_idx = random.choice(forgotten_idx)
                pos_example = pt_ds[pos_idx]
                label = fgt_mat[ocl_ds_idx, pos_idx]
                if self.config.fpd.binarilize_labels:
                    label = 1 if label > 0 else 0
                examples.append({
                    'input_ids': pos_example['input_ids'],
                    'input_ids_ocl': ocl_example['input_ids'],
                    'input_ids_pt': pos_example['input_ids'],
                    'forget_label': label,
                    'pt_idx': pos_idx,
                    'ocl_ds_idx': ocl_ds_idx,
                    'ocl_ex_idx': ocl_idx
                })
            if len(non_forgotten_idx) > 0:
                neg_idx = random.choice(non_forgotten_idx)
                neg_example = pt_ds[neg_idx]
                label = fgt_mat[ocl_ds_idx, neg_idx]
                if self.config.fpd.binarilize_labels:
                    label = 1 if label > 0 else 0
                examples.append({
                    'input_ids': neg_example['input_ids'],
                    'input_ids_ocl': ocl_example['input_ids'],
                    'input_ids_pt': neg_example['input_ids'],
                    'forget_label': label,
                    'pt_idx': neg_idx,
                    'ocl_ds_idx': ocl_ds_idx,
                    'ocl_ex_idx': ocl_idx
                })
        batch = self.collator(examples)
        return batch

    def sample_episode_task_level(self, split, bs):
        ocl_dss, pt_ds = getattr(self, '{}_ocl_dss'.format(split)), self.pt_ds
        fgt_mat = getattr(self, '{}_mat'.format(split))
        ocl_ds_num, pt_ex_num = len(ocl_dss), len(pt_ds)

        examples = []
        for b in range(bs):
            pt_idx, ocl_ds_idx = random.choice(range(pt_ex_num)), random.choice(range(ocl_ds_num))
            ocl_ds = ocl_dss[ocl_ds_idx]
            ocl_idx = random.choice(range(len(ocl_ds)))

            pt_example, ocl_example = pt_ds[pt_idx], ocl_ds[ocl_idx]
            label = fgt_mat[ocl_ds_idx, pt_idx]
            if self.config.fpd.binarilize_labels:
                label = 1 if label > 0 else 0

            example = {
                'input_ids': pt_example['input_ids'],
                'input_ids_ocl': ocl_example['input_ids'],
                'input_ids_pt': pt_example['input_ids'],
                'forget_label': label,
                'pt_idx': pt_idx,
                'ocl_ds_idx': ocl_ds_idx,
                'ocl_ex_idx': ocl_idx
            }
            examples.append(example)
        batch = self.collator(examples)
        return batch

    def get_ocl_dataloader_concat(self, split, batch_size):
        ocl_dss = getattr(self, '{}_ocl_dss'.format(split))
        concat_ds = ConcatDataset(ocl_dss)
        loader = DataLoader(concat_ds, batch_size, shuffle=False, collate_fn=self.collator)
        return loader

    def expand_scores(self, ocl_dss, scores):
        score_expand = []
        for ocl_idx in range(len(ocl_dss)):
            score_expand.extend([scores[ocl_idx]] * len(ocl_dss[ocl_idx]))
        score_expand = np.stack(score_expand)
        score_expand = torch.from_numpy(score_expand)
        return score_expand

    def get_label_grid_and_base_error_binarilize(self, split):
        ocl_error_dss = getattr(self, f'{split}_ocl_dss')
        pt_ds = getattr(self, f'{split}_pt_ds')
        scores = torch.from_numpy(getattr(self, f'{split}_mat'))
        expand_scores = self.expand_scores(ocl_error_dss, scores)
        label_grid = torch.where(expand_scores > 0, 1, 0)
        base_error_mask = torch.zeros(len(pt_ds)).bool()
        #for ocl_error_idx in range(len(ocl_error_ds)):
        #    label_grid[ocl_error_idx, gt_forgets[ocl_error_idx]] = 1
        return label_grid, base_error_mask

    def evaluate_metrics(self, fgt_label_grid, preds_grid, base_error_mask):
        f1s = []
        ps, rs = [], []
        for ocl_error_idx in range(fgt_label_grid.size(0)):
            valid_label = fgt_label_grid[ocl_error_idx, ~base_error_mask].detach().cpu().numpy()
            valid_pred = preds_grid[ocl_error_idx, ~base_error_mask].detach().cpu().numpy()
            #print(valid_label, valid_pred, valid_pred.shape, valid_label.shape, ~base_error_mask)
            f1 = f1_score(valid_label, valid_pred)
            f1s.append(f1)
            p, r = precision_score(valid_label, valid_pred), recall_score(valid_label, valid_pred)
            ps.append(p)
            rs.append(r)
        ret = {
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'p_mean': np.mean(ps),
            'p_std': np.std(ps),
            'r_mean': np.mean(rs),
            'r_std': np.std(rs),
            'task': self.ocl_task
        }
        return ret

    def save_preds(self, preds_grid, prob_grid, base_error_mask, output_dir, split):
        os.makedirs(output_dir,exist_ok=True)
        ret = {}
        all_idxs = np.arange(preds_grid.shape[1])[~base_error_mask]
        for ocl_error_idx in range(preds_grid.shape[0]):
            valid_pred = preds_grid[ocl_error_idx, ~base_error_mask].detach().cpu().numpy().astype(bool)
            ret[ocl_error_idx] = all_idxs[valid_pred].tolist()
        with open(os.path.join(output_dir, 'forgets.pkl'), 'wb') as wf:
            pickle.dump(ret, wf)
        #save_dataset(getattr(self,f'{split}_ocl_error_ds'), os.path.join(output_dir, f'ocl_error_ds.csv'))
        #save_dataset(getattr(self,f'{split}_pt_ds'), os.path.join(output_dir, 'concat_pt_ds.csv'))

    def save_raw_scores(self, rep_prods, output_dir, split):
        if torch.is_tensor(rep_prods):
            rep_prods = rep_prods.cpu().numpy()
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f'rep_prods_{split}.npy'), rep_prods)


    def save_pred_logits(self, all_pred_logits, all_pred_logits_idxs, output_dir, split):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'pred_logits_{split}.pkl'), 'wb') as wf:
            pickle.dump({
                'pred_logits': all_pred_logits,
                'pred_logits_idxs': all_pred_logits_idxs
            }, wf)

    def get_forget_freqs(self, split):
        gt_forgets = getattr(self, f'{split}_gt_forgets')
        pt_ds = getattr(self, f'{split}_pt_ds')
        freqs = defaultdict(int)
        for ocl_error_idx, gts in gt_forgets.items():
            for gt in gts:
                freqs[gt] += 1
        for idx in range(len(pt_ds)):
            if idx not in freqs:
                freqs[idx] = 0
        total = len(gt_forgets)
        return freqs, total

    def predict_thres_based(self, freqs, perc):
        freq_vs = sorted([_ for _ in freqs.values()])
        thres = np.percentile(freq_vs, perc)
        preds = [k for k, v in freqs.items() if v > thres]
        # print(len(preds), thres)
        return preds

    def get_fixed_bias_terms(self, split):
        fgt_freqs, total = self.get_forget_freqs(split)
        # tranlate to prob / odds
        prior_odds = {k: (v + 1) / (total - v + 1) for k, v in fgt_freqs.items()}

        return prior_odds

    def get_all_priors(self, split):
        priors = getattr(self, f'{split}_priors')
        if priors is None:
            return None
        else:
            pt_ds = getattr(self, f'{split}_pt_ds')
            prior_arr = [priors[idx] for idx in range(len(pt_ds))]
            prior_arr = torch.FloatTensor(prior_arr)
            return prior_arr


class DataCollatorWithPaddingStrForFpd(DataCollatorWithPadding):
    def pad_logits_or_idxs(self, tensor_list):
        max_len = max([len(x) for x in tensor_list])
        dim_size = tensor_list[0].size(-1)
        out = torch.zeros(len(tensor_list), max_len, dim_size, dtype=tensor_list[0].dtype)
        for i, tensor in enumerate(tensor_list):
            out[i, :len(tensor)] = tensor
        return out

    def __call__(self, features):
        features_non_str = []
        features_str = []

        special_feat_names = ['pt_logits_ss', 'pt_logits_idxs', 'ocl_update_logits','pt_logits_after_ss','pt_logits_after_idxs']
        special_feats = {k: [] for k in special_feat_names}

        for feature in features:
            dic, dic2 = {}, {}
            for k, v in feature.items():
                if k in special_feat_names:
                    special_feats[k].append(v)
                elif type(v) is not str:
                    dic[k] = v
                else:
                    dic2[k] = v
            features_non_str.append(dic)
            features_str.append(dic2)

        batch = self.tokenizer.pad(
            features_non_str,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        for k, vs in special_feats.items():
            if vs:
                out = self.pad_logits_or_idxs(vs)
                batch[k] = out

        for dic in features_str:
            for k, v in dic.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        return batch


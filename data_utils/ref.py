
import os
from datasets import Dataset
import json
import numpy as np

import logging
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import ConcatDataset

EMOTION_LABELS = ['joy','anger','sadness','disgust','fear','trust','surprise','love','noemo','confusion','anticipation','shame','guilt',
                    'valence','arousal','dominance']
IGNORE_ID = -100


class EmotionDataManager(GLUEDataManager):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

    def update_model_config(self, model_config):
        model_config.num_labels = len(EMOTION_LABELS)
        model_config.label2id = {v: i for i,v in enumerate(EMOTION_LABELS)}
        model_config.id2label = {i: label for label, i in model_config.label2id.items()}

        tasks = model_config.dataset_name.split()
        train_examples = self.load_examples('train', *tasks)
        model_config.freq_dist = self.get_freq_dist_examples(train_examples['label'])


    def load_dataset(self, model_config, partition=None):
        # raw_datasets = load_dataset('glue', model_config.dataset_name,
        #         cache_dir=os.path.join(self.config.hf_datasets_cache_dir, 'datasets'))
        tasks = model_config.dataset_name.split()
        split_examples = self.load_all_splits(*tasks)
        #model_config.freq_dist = self.get_freq_dist_examples(split_examples['train']['label'])

        def preprocess_function(examples):
            # Tokenize the texts
            result = self.tokenizer(examples['sentence'], max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            result['label'] = examples['label']
            return result

        max_seq_length = model_config.max_seq_length
        split_datasets = {split: Dataset.from_dict(examples) for split, examples in split_examples.items()}
        for split, dataset in split_datasets.items():
            cache_file_name = os.path.join(self.config.hf_datasets_cache_dir,
                "{}-{}-{}-{}".format('_'.join(tasks), split, self.tokenizer.name_or_path.split('/')[-1], get_dataset_digest(dataset)))
            dataset = dataset.map(preprocess_function, batched=True)
            split_datasets[split] = dataset

        train_dataset = split_datasets['train']
        eval_dataset = split_datasets["dev"]
        test_dataset = split_datasets['test']

        if model_config.train_subset_n != -1:
            if len(train_dataset) > model_config.train_subset_n:
                rng = np.random.default_rng(self.config.seed) if model_config.train_subset_seed is None else np.random.default_rng(model_config.train_subset_seed)
                idxs = rng.choice(len(train_dataset), model_config.train_subset_n, replace=False).tolist()
                train_dataset = Subset(train_dataset, idxs)
                logging.info('Subsampled {} examples for {}, {}'.format(model_config.train_subset_n, model_config.dataset_name, idxs[:5]))
            else:
                logging.info('Not subsampled {} ({}<{})'.formafreqt(model_config.dataset_name, len(train_dataset), model_config.train_subset_n))

        if model_config.partition != -1 and self.config.partition.method is not None:
            if self.config.partition.method == 'iid':
                train_dataset = self.get_partition_subset_iid(model_config, train_dataset)
            else:
                train_dataset = self.get_partition_subset_label_niid(model_config, train_dataset)

        #metrics_func = load_metric('glue', model_config.dataset_name,
        #        cache_dir=os.path.join(self.config.hf_datasets_cache_dir, 'metrics'))
        return train_dataset, eval_dataset, test_dataset

    def load_all_splits(self, *filter_tasks):
        ret = {}
        for split in ['train','dev', 'test']:
            ret[split] = self.load_examples(split, *filter_tasks)
        return ret

    def get_freq_dist_examples(self, labels):
        labels = np.array(labels)
        dist = {}
        for label_id, label_name in enumerate(EMOTION_LABELS):
            arr = labels[:,label_id]
            zeros, ones = (arr==0).sum(), (arr==1).sum()
            if zeros != 0 or ones != 0:
                dist[label_id] = {0: int(zeros), 1: int(ones)}
        return dist

    def get_freq_dist(self, dataset):
        dist = {}
        for item in dataset:
            for label_id, label_name in enumerate(EMOTION_LABELS):
                if item['label'][label_id] != IGNORE_ID:
                    if label_id not in dist:
                        dist[label_id] = {0: 0, 1: 0}
                    v = item['label'][label_id]
                    dist[label_id][v] += 1
        return dist

    def load_examples(self, split, *filter_tasks):
        lines = []
        for task in filter_tasks:
            with open(os.path.join(self.config.resource_dir, 'emotion_splits/{}/{}.jsonl'.format(task, split))) as f:
                lines_partial = f.readlines()
                lines.extend(lines_partial)
        data = [json.loads(x) for x in lines]
        dic = {'sentence':  [], 'label': []}
        for example in data:
            label = [IGNORE_ID] * len(EMOTION_LABELS)
            for ename, ev in example['emotions'].items():
                ev = self.normalize_label(ev)
                label[EMOTION_LABELS.index(ename)] = ev
            for vname, vv in example['VAD'].items():
                vv = self.normalize_label(vv)
                label[EMOTION_LABELS.index(vname)] = vv
            dic['label'].append(label)
            dic['sentence'].append(example['text'])
        return dic

    def normalize_label(self, v):
        if v is None:
            return IGNORE_ID
        else:
            return 1 if v > 0.1 else 0

    def get_metrics_func_single(self, model_config):
        def compute_metrics(p):
            result = {}
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # [N, L]
            labels = p.label_ids

            for label_idx, label_name in enumerate(EMOTION_LABELS):
                mask = p.label_ids[:, label_idx] != IGNORE_ID
                if mask.sum() != 0:
                    y_score, y = preds[mask, label_idx], labels[mask, label_idx]
                    y_hat = y_score >= 0.5
                    f1 = f1_score(y, y_hat)
                    prec, recall = precision_score(y, y_hat), recall_score(y, y_hat)
                    result[f'{label_name}_f1'], result[f'{label_name}_prec'], result[f'{label_name}_recall'] = f1, prec, recall

            result["macro_f1"] = np.mean([v for k,v in result.items() if k.endswith('_f1')])
            result["key_score"] = result['macro_f1']
            return result
        return compute_metrics



    # def get_metrics_func(self, model_config):
    #     if self.config.mtl:
    #         model_config = next(iter(vars(self.config.local_models._models).values()))
    #     return self.get_metrics_func_single(model_config)

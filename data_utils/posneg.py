from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
import pickle
import json
import random

class NLIPosNegDataManager:
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.nli_id2label = ['entailment','neutral','contradiction']

    def update_model_config(self, config):
        self.config.num_labels = 2
        labels = ['not_affected', 'affected']
        self.config.label2id = config.label2id = {v: i for i,v in enumerate(labels)}
        self.config.id2label = config.id2label = {i: label for label, i in config.label2id.items()}

    def create_examples(self, raw_records):
        input_texts = []
        labels = []

        for record in raw_records:
            input_items = [record['ocl_premise'], record['ocl_hypo'], self.nli_id2label[record['ocl_label']],
                           record['pos_premise'], record['pos_hypo'], self.nli_id2label[record['pos_label']]]
            sep = ' ' + self.tokenizer.sep_token + ' '
            text = sep.join(input_items)
            input_texts.append(text)
            labels.append(1)

            input_items = [record['ocl_premise'], record['ocl_hypo'], self.nli_id2label[record['ocl_label']],
                           record['neg_premise'], record['neg_hypo'], self.nli_id2label[record['neg_label']]]
            text = sep.join(input_items)
            input_texts.append(text)
            labels.append(0)
        return input_texts, labels

    def load_dataset(self, config):
        dataset_path = config.posneg_file
        split_paths = {
            split: dataset_path.replace('SPLIT', split) for split in ['train','dev','test']
        }
        split_datasets = {}
        for split_name, path in split_paths.items():
            with open(path,'r') as f:
                examples = json.load(f)

            if len(examples) > getattr(config, f'max_{split_name}_example'):
                # fixed subsample
                examples = random.Random(0).sample(examples, getattr(config, f'max_{split_name}_example'))

            input_texts, labels = self.create_examples(examples)
            inputs = self.tokenizer(input_texts, max_length=self.config.max_seq_length, truncation=True)
            inputs['labels'] = labels
            ds = Dataset.from_dict(inputs)
            split_datasets[split_name] = ds
        return split_datasets['train'], split_datasets['dev'], split_datasets['test']

    def get_metrics_func(self):
        def compute_metrics(p):
            result = {}
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # [N, L]
            labels = p.label_ids
            pred_labels = preds.argmax(-1)
            f1 = f1_score(labels, pred_labels, average='macro')
            acc = accuracy_score(labels, pred_labels)
            result["macro_f1"] = f1
            result["acc"] = acc
            return result
        return compute_metrics



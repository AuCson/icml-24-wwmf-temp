from torch.utils.data import Dataset
import numpy as np
import os
from olmo.data.memmap_dataset import MemMapDataset
from .utils import truncate_prefix
from .mmlu import MMLUHelper
from .bbh import BBHHelper
from .utils import apply_chat_template, apply_chat_template_for_generation
import json
import pickle
import datasets
import random

def get_paloma_task2files(base_dir):
    ret = {}
    for bd, dirs, files in os.walk(base_dir):
        dir_parts = bd.split('/')
        task, split = None, None
        if len(dir_parts) >= 2:
            task, split = dir_parts[-2], dir_parts[-1]
        for file in files:
            if file.endswith('.npy'):
                if task not in ret:
                    ret[task] = {}
                if split not in ret[task]:
                    ret[task][split] = []
                ret[task][split].append(os.path.join(bd, file))
    return ret

def get_dolma_files(base_dir):
    files = sorted(os.listdir(base_dir))
    paths = [os.path.join(base_dir, x) for x in files]
    return paths

class PalomaDataset(Dataset):
    def __init__(self, config, tokenizer, examples=None):
        self.config = config
        self.tokenizer = tokenizer
        #self.collator = collator
        self.examples = examples

    def load_from_token_npy(self, file):
        tokens = np.load(file)
        examples = []

        # truncate
        max_length = self.config.max_input_length
        start = 0
        while start < len(tokens):
            stop = min(start + max_length, len(tokens))
            examples.append(tokens[start:stop])
            start = stop

        return examples

    def load_all_from_dir(self, base_dir, filter_task=None, filter_split=None):
        if self.examples is not None:
            raise ValueError('Dataset already loaded')
        all_examples = []
        task2files = get_paloma_task2files(base_dir)
        for task in task2files:
            if filter_task and task not in filter_task:
                continue
            for split in task2files[task]:
                if filter_split and split not in filter_split:
                    continue
                files = task2files[task][split]

                for file in files:
                    raw_examples = self.load_from_token_npy(file)
                    examples = [{'input_ids': x, 'task_name': task, 'split': split} for x in raw_examples]
                    all_examples.extend(examples)
        self.examples = all_examples

    def get_ds_by_task(self):
        task2examples = {}
        for example in self.examples:
            task = example['task_name']
            if task not in task2examples:
                task2examples[task] = []
            task2examples[task].append(example)
        task2ds = {}
        for task, task_examples in task2examples.items():
            task2ds[task] = PalomaDataset(self.config, self.tokenizer, task_examples)
        return task2ds

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example

    def __len__(self):
        return len(self.examples)

class DolmaDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        #self.collator = collator
        self.mm_ds = None
        self.max_length = config.max_input_length

    def load_all_from_dir(self, base_dir, start, stop):
        paths = get_dolma_files(base_dir)
        mm_ds = MemMapDataset(*paths[start:stop], chunk_size=self.config.max_input_length, memmap_dtype=np.int64)
        self.mm_ds = mm_ds

    def __getitem__(self, idx):
        raw_example = self.mm_ds[idx]
        input_ids = raw_example['input_ids']
        #FIXME
        input_ids[0] = 0
        #if len(raw_example) < self.max_length:
        #    pad = np.zeros(self.max_length - len(raw_example)).astype(raw_example['input_ids'])
        #    input_ids = np.concatenate([pad, input_ids])
        example = {
            'input_ids': input_ids,
            'task_name': 'dolma',
            #'labels': input_ids
        }
        return example

    def __len__(self):
        return len(self.mm_ds)



class SFTDataset(Dataset):
    def __init__(self, config, tokenier, input_texts, task_names, indexes):
        self.config = config
        self.tokenizer = tokenier
        self.input_texts = input_texts
        self.input_encoding = truncate_prefix(tokenier, input_texts, self.config.max_input_length)
        self.indexes = indexes
        self.task_names = task_names

    @classmethod
    def from_mmlu(cls, tasks, split, config, tokenizer, skip_encoding=False):
        all_examples = []
        all_task_names = []
        for task in tasks:
            mmlu_helper = MMLUHelper(config, task)
            answer_type = config.mmlu.answer_type
            cot = config.mmlu.is_cot
            few_shot = config.mmlu.is_few_shot
            prompt = mmlu_helper.get_prompt(task, cot=cot, answer_type=answer_type, is_few_shot=few_shot)
            examples = mmlu_helper.create_examples(split, prompt, cot=cot, answer_type=answer_type, example_format='lm')
            task_names = [example[-1] for example in examples]

            all_examples.extend(examples)
            all_task_names.extend(task_names)
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_bbh(cls, tasks, split, config, tokenizer, skip_encoding=False):
        all_examples = []
        all_task_names = []
        for task in tasks:
            mmlu_helper = BBHHelper(config, task)
            cot = config.bbh.is_cot
            examples = mmlu_helper.create_examples(split, cot=cot,  example_format='lm')
            task_names = [example[-1] for example in examples]

            all_examples.extend(examples)
            all_task_names.extend(task_names)
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_flan_by_task(cls, tasks, split, config, tokenizer, skip_encoding=False):
        all_examples = []
        all_task_names = []
        if split == 'dev':
            split_name = 'validation'
        else:
            split_name = split

        for task in tasks:
            with open(os.path.join(config.flan_by_task_dir, '{}_{}.json'.format(task, split_name))) as f:
                data = json.load(f)
            examples = [[x['inputs'], x['targets'], x['task']] for x in data]
            all_examples.extend(examples)
            all_task_names.extend([example[-1] for example in examples])

        if getattr(config, 'max_flan_example', -1) > 0:
            print('Max flan example is {}'.format(config.max_flan_example))

            all_examples = all_examples[:config.max_flan_example]
            all_task_names = all_task_names[:config.max_flan_example]
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_tulu_train(cls, tasks, split, config, tokenizer, skip_encoding=False):
        ds = datasets.load_dataset("allenai/tulu-v2-sft-mixture")
        all_examples = []
        for example in ds['train']:
            task = example['dataset'].split('.')[0]
            if task in tasks:
                all_examples.append([
                    example['messages'][0]['content'],
                    example['messages'][1]['content'],
                    task
                ])

        all_task_names = [_[-1] for _ in all_examples]
        if getattr(config, 'max_tulu_train_example', -1) > 0:
            print('Max tulu_train example is {}'.format(config.max_tulu_train_example))
            all_examples = all_examples[:config.max_flan_example]
            all_task_names = all_task_names[:config.max_flan_example]

        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_tulu(cls, tasks, split, config, tokenizer, skip_encoding=False):
        with open(config.tulu.path) as f:
            raw_examples = json.load(f)
        all_examples = [
            [example['messages'][0]['content'],
             example['messages'][1]['content'],
             example['dataset'].split('.')[0]] for example in raw_examples
        ]
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        all_task_names = [_[-1] for _ in all_examples]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_truthful_qa(cls, tasks, split, config, tokenizer, skip_encoding=False):
        ds = datasets.load_dataset('truthful_qa', 'generation')
        all_examples = []
        for example in ds['validation']:
            task = example['category'].split(':')[0]
            if task in tasks:
                all_examples.append([
                    example['question'],
                    example['best_answer'],
                    example['category']
                ])
        all_task_names = [_[-1] for _ in all_examples]
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_dolma_sample(cls, tasks, split, config, tokenizer, skip_encoding=False):
        with open(config.dolma.sample_path,'rb') as f:
            raw_examples = pickle.load(f)
        all_examples = [
            [example['text'], '{}_{}'.format(example['example_id'], example['chunk_id'])] for example in raw_examples
        ]
        input_texts = [x[0] for x in all_examples]
        all_task_names = [x[-1] for x in all_examples]
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes)
        return ds

    @classmethod
    def from_auto(cls, ds_category, **kwargs):
        if ds_category == 'mmlu':
            return cls.from_mmlu(**kwargs)
        elif ds_category == 'bbh':
            return cls.from_bbh(**kwargs)
        elif ds_category == 'tulu':
            return cls.from_tulu(**kwargs)
        elif ds_category == 'truthful_qa':
            return cls.from_truthful_qa(**kwargs)
        elif ds_category == 'tulu_train':
            return cls.from_tulu_train(**kwargs)
        elif ds_category == 'dolma_sample':
            return cls.from_dolma_sample(**kwargs)
        elif ds_category == 'flan':
            return cls.from_flan_by_task(**kwargs)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        input_ids = self.input_encoding['input_ids'][idx]
        task_name = self.task_names[idx]

        example = {
            'input_ids': input_ids,
            'task_name': task_name
        }
        return example

    def __len__(self):
        return len(self.input_encoding['input_ids'])


class SFTExampleOnlyDataset(Dataset):
    def __init__(self, examples, is_lm=False):
        self.examples = examples
        self.is_lm = is_lm

    @classmethod
    def from_mmlu(cls, tasks, split, config):
        all_examples = []
        for task in tasks:
            mmlu_helper = MMLUHelper(config, task)
            answer_type = config.mmlu.answer_type
            cot = config.mmlu.is_cot
            few_shot = config.mmlu.is_few_shot
            prompt = mmlu_helper.get_prompt(task, cot=cot, answer_type=answer_type, is_few_shot=few_shot)
            examples = mmlu_helper.create_examples(split, prompt, cot=cot, answer_type=answer_type, example_format='lm')
            task_names = [example[-1] for example in examples]
            all_examples.extend(examples)
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_bbh(cls, tasks, split, config):
        all_examples = []
        for task in tasks:
            bbh_helper = BBHHelper(config, task)

            cot = config.bbh.is_cot
            examples = bbh_helper.create_examples(split,  cot=cot, example_format='lm')
            task_names = [example[-1] for example in examples]
            all_examples.extend(examples)
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_tulu(cls, config):
        with open(config.tulu.path) as f:
            raw_examples = json.load(f)
        all_examples = [
            [example['messages'][0]['content'],
             example['messages'][1]['content'],
             example['dataset'].split('.')[0]] for example in raw_examples
        ]
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_dolma(cls, config):
        with open(config.dolma.sample_path,'rb') as f:
            raw_examples = pickle.load(f)
        all_examples = [
            [example['text'], '{}_{}'.format(example['example_id'], example['chunk_id'])] for example in raw_examples
        ]
        ds = cls(all_examples, is_lm=True)
        return ds

    @classmethod
    def from_truthful_qa(cls, config, tasks):
        ds = datasets.load_dataset('truthful_qa', 'generation')
        all_examples = []
        for example in ds['validation']:
            task = example['category'].split(':')[0]
            if task in tasks:
                all_examples.append([
                    example['question'],
                    example['best_answer'],
                    example['category']
                ])
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_tulu_train(cls, config, tasks):
        ds = datasets.load_dataset("allenai/tulu-v2-sft-mixture")
        all_examples = []
        for example in ds['train']:
            task = example['dataset'].split('.')[0]
            if task in tasks:
                all_examples.append([
                    example['messages'][0]['content'],
                    example['messages'][1]['content'],
                    task
                ])
        ds = cls(all_examples)
        return ds


    @classmethod
    def from_auto(cls, ds_category, **kwargs):
        if ds_category == 'mmlu':
            return cls.from_mmlu(**kwargs)
        elif ds_category == 'bbh':
            return cls.from_bbh(**kwargs)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def get_chat_or_raw_examples(self, include_gt, tokenizer):
        if self.is_lm:
            input_texts = [x[0] for x in self.examples]
        else:
            if include_gt:
                input_texts = apply_chat_template(self.examples, tokenizer)
            else:
                input_texts = apply_chat_template_for_generation(self.examples, tokenizer)
        return input_texts

    def get_gt_answers(self):
        return [example[1] for example in self.examples]
from .p3 import P3Dataset
from torch.utils.data import Dataset
import os
import json
from transformers import DataCollatorWithPadding
import random

T5PROMPT = 'Exam: please fill in the blank. {} ___'

class CounterFactDataset(P3Dataset):
    def __init__(self, config, examples, input_encoding, answer_encoding):
        self.config = config
        self.examples = examples
        self.input_encoding = input_encoding
        self.answer_encoding = answer_encoding

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = {k: v for k,v in self.examples[idx].items()}
        example['input_ids'] = self.input_encoding['input_ids'][idx]
        example['attention_mask'] = self.input_encoding['attention_mask'][idx]
        example['labels'] = self.answer_encoding['input_ids'][idx]
        return example

class CounterFactDatasetHelper:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.raw_pretrain_examples, self.raw_ocl_examples, self.raw_dev_examples = None, None, None
        self.load_examples()
        self.collator = DataCollatorWithPadding(tokenizer)
        self.random = random.Random(0)

    def load_examples(self):
        with open(os.path.join(self.config.cf.data_dir, 'counterfact_pretrain.json')) as f:
            self.raw_pretrain_examples = json.load(f)
        with open(os.path.join(self.config.cf.data_dir, 'counterfact_ocl.json')) as f:
            self.raw_ocl_examples = json.load(f)
        with open(os.path.join(self.config.cf.data_dir, 'counterfact_dev.json')) as f:
            self.raw_dev_examples = json.load(f)

    def construct_input_and_ans(self, raw_examples, true_output=True):
        output_examples = []
        para_output_examples = []
        nb_output_examples = []

        for example_id, raw_example in enumerate(raw_examples):
            raw_input = raw_example['requested_rewrite']['prompt'].format(raw_example['requested_rewrite']['subject'])
            prompted_input = T5PROMPT.format(raw_input)
            if true_output:
                answer = raw_example['requested_rewrite']['target_true']['str']
            else:
                answer = raw_example['requested_rewrite']['target_new']['str']
            para_inputs = []
            for para in raw_example['paraphrase_prompts']:
                para_inputs.append(T5PROMPT.format(para))
            nb_inputs = []
            for nb in raw_example['neighborhood_prompts']:
                nb_inputs.append(T5PROMPT.format(nb))
            case_id = raw_example['case_id']

            output_examples.append({
                'original_input': prompted_input,
                'original_answers': answer,
                'case_id': case_id,
                'example_id': example_id,
                'task_name': 'cf'
            })

            para_output_examples.extend([{
                'original_input': para_input,
                'original_answers': answer,
                'case_id': case_id,
                'example_id': example_id,
                'task_name': 'cf'
            } for para_input in para_inputs])

            nb_output_examples.extend([{
                'original_input': nb_input,
                'original_answers': answer,
                'case_id': case_id,
                'example_id': example_id,
                'task_name': 'cf'
            } for nb_input in nb_inputs])

        return output_examples, para_output_examples, nb_output_examples

    def construct_episodic_input_ans(self, raw_examples, true_output=True):
        episodic_examples = []

        for example_id, raw_example in enumerate(raw_examples):
            raw_input = raw_example['requested_rewrite']['prompt'].format(raw_example['requested_rewrite']['subject'])
            prompted_input = T5PROMPT.format(raw_input)
            if true_output:
                answer = raw_example['requested_rewrite']['target_true']['str']
            else:
                answer = raw_example['requested_rewrite']['target_new']['str']
            para_inputs = []
            for para in raw_example['paraphrase_prompts']:
                para_inputs.append(T5PROMPT.format(para))
            nb_inputs = []
            for nb in raw_example['neighborhood_prompts']:
                nb_inputs.append(T5PROMPT.format(nb))
            case_id = raw_example['case_id']

            training_example = {
                'original_input': prompted_input,
                'original_answers': answer,
                'case_id': case_id,
                'example_id': example_id,
                'task_name': 'cf',
                'example_type': 'training'
            }

            para_examples = [{
                'original_input': para_input,
                'original_answers': answer,
                'case_id': case_id,
                'example_id': example_id,
                'task_name': 'cf',
                'example_type': 'para_{}'.format(n)
            } for n, para_input in enumerate(para_inputs)]

            nb_examples = [{
                'original_input': nb_input,
                'original_answers': answer,
                'case_id': case_id,
                'example_id': example_id,
                'task_name': 'cf',
                'example_type': 'nb_{}'.format(n),
            } for n, nb_input in enumerate(nb_inputs)]

            episodic_examples.append({
                'training': training_example,
                'paras': para_examples,
                'nbs': nb_examples
            })

        return episodic_examples

    def construct_abl_stream(self, episodic_examples):
        abl_episodic_examples = []
        all_training_examples = [x['training'] for x in episodic_examples]
        self.random.shuffle(all_training_examples)

        for i, ep in enumerate(episodic_examples):
            abl_episodic_examples.append({
                'training': all_training_examples[i],
                'paras': ep['paras'],
                'nbs': ep['nbs']
            })
        return abl_episodic_examples


    def encode_examples(self, examples):
        inputs = [x['original_input'] for x in examples]
        answers = [x['original_answers'] for x in examples]
        input_encoding = self.tokenizer.batch_encode_plus(inputs, padding='longest',max_length=self.config.max_input_length)
        answer_encoding = self.tokenizer.batch_encode_plus(answers, padding='longest', max_length=self.config.max_output_length)
        return input_encoding, answer_encoding

    def encode_examples_to_batch(self, examples):
        input_encoding, answer_encoding = self.encode_examples(examples)
        #batch = {}
        #batch['input_ids'] = input_encoding['input_ids']
        #batch['attention_mask'] = input_encoding['attention_mask']
        #batch['labels'] = answer_encoding['input_ids']
        enc_examples = []
        for b in range(len(examples)):
            enc_examples.append({
                'input_ids': input_encoding['input_ids'][b],
                'attention_mask': input_encoding['attention_mask'][b],
                'labels': answer_encoding['input_ids'][b]
            })

        batch = self.collator(enc_examples)
        return batch

    def raw_examples_to_ds(self, examples):
        input_encoding, answer_encoding = self.encode_examples(examples)
        ds = CounterFactDataset(self.config, examples, input_encoding, answer_encoding)
        return ds

    def random_correct_fact_ocl(self):
        pretraining_examples, _, _ = self.construct_input_and_ans(self.raw_pretrain_examples)
        ocl_examples, _, _ = self.construct_input_and_ans(self.raw_ocl_examples)
        pretrain_ds = self.raw_examples_to_ds(pretraining_examples)
        ocl_ds = self.raw_examples_to_ds(ocl_examples)
        return pretrain_ds, ocl_ds

    def random_incorrect_fact_ocl(self):
        pretraining_examples, _, _ = self.construct_input_and_ans(self.raw_pretrain_examples, true_output=True)
        ocl_examples, _, _ = self.construct_input_and_ans(self.raw_ocl_examples, true_output=False)
        pretrain_ds = self.raw_examples_to_ds(pretraining_examples)
        ocl_ds = self.raw_examples_to_ds(ocl_examples)
        return pretrain_ds, ocl_ds

    def ocl_episodes(self, use_abl=False, use_true_output=False):
        ret = self.construct_episodic_input_ans(self.raw_ocl_examples, true_output=use_true_output)
        if use_abl:
            abl_stream = self.construct_abl_stream(ret)
            return abl_stream
        return ret

# class DataCollatorWithPaddingStr(DataCollatorWithPadding):
#     def __call__(self, features):
#         features_non_str = []
#         features_str = []
#         for feature in features:
#             dic, dic2 = {}, {}
#             for k, v in feature.items():
#                 if type(v) is not str:
#                     dic[k] = v
#                 else:
#                     dic2[k] = v
#             features_non_str.append(dic)
#             features_str.append(dic2)
#
#         batch = self.tokenizer.pad(
#             features_non_str,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         if "label" in batch:
#             batch["labels"] = batch["label"]
#             del batch["label"]
#         if "label_ids" in batch:
#             batch["labels"] = batch["label_ids"]
#             del batch["label_ids"]
#
#         for dic in features_str:
#             for k, v in dic.items():
#                 if k not in batch:
#                     batch[k] = []
#                 batch[k].append(v)
#
#         return batch

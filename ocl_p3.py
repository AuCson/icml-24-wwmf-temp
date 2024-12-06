from memory_utils.multi_dataset import MultiTaskMemory
from data_utils.p3 import P3Dataset, load_bg_train_ds, load_ocl_dss
import os
from trainer.my_trainer import MyTrainer
from trainer.utils import trim_batch, DataCollatorWithPaddingStr
from transformers import BartForConditionalGeneration, AutoTokenizer, TrainingArguments
from utils.config import merge_config_into_args, load_configs
import argparse
import logging
import json
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
from utils.analysis_tools import run_evaluate

logger = logging.getLogger()

def load_background_train_dss(config, tokenizer, skip_encoding=True):
    dss_names = config.pt_tasks
    pt_dss = {}
    for ds_name in dss_names:
        path = os.path.join(config.pretrain_ds_dir, '{}.json'.format(ds_name))
        ds = P3Dataset.from_file(path, ds_name, config, tokenizer, skip_encoding=skip_encoding)
        pt_dss[ds_name] = ds
    return pt_dss

def load_background_eval_dss(config, tokenizer):
    dss_names = config.pt_tasks
    dev_dss, test_dss = {}, {}
    for ds_name in dss_names:
        path = os.path.join(config.ocl_ds_dir, ds_name, 'validation-1000.json'.format(ds_name))
        ds = P3Dataset.from_file(path, ds_name, config, tokenizer)
        dev_dss[ds_name] = ds

        path = os.path.join(config.ocl_ds_dir, ds_name, 'test-1000.json'.format(ds_name))
        ds = P3Dataset.from_file(path, ds_name, config, tokenizer)
        test_dss[ds_name] = ds
    return dev_dss, test_dss


def get_ocl_eval_func(config, model, trainer, eval_ds, task, task_id):
    # evaluate on ocl task
    def eval_func(**kwargs):
        all_preds, all_gts = run_evaluate(config, model, trainer, eval_ds)
        scores = eval_ds.compute_metrics(all_gts, all_preds)
        logger.info('{}\t{}\tOCL\t{}'.format(task, json.dumps(scores), task_id))
    return eval_func

def run_pipeline(args):
    config = load_configs(*args.config_files, templates=args.templates)

    model =  BartForConditionalGeneration.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    data_collator = DataCollatorWithPaddingStr(tokenizer)

    training_args = TrainingArguments(output_dir=config.output_dir)
    merge_config_into_args(config, training_args)


    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir,'log.txt')))

    trainer = MyTrainer(
        model=model, args=training_args, train_dataset=None, eval_dataset=None,
        tokenizer=tokenizer, data_collator=data_collator, memory=None, config=config
    )

    # load all pt eval dss
    pt_dev_dss, pt_test_dss = load_background_eval_dss(config, tokenizer)
    ocl_train_dss, ocl_dev_dss, ocl_test_dss = load_ocl_dss(config, tokenizer)

    pt_eval_dss = pt_dev_dss
    ocl_eval_dss = ocl_dev_dss

    # initial evaluation on pt datasets

    if args.do_init_eval_pt:
        logger.info('--init eval pt--')
        for task_name, pt_eval_ds in pt_eval_dss.items():
            all_preds, all_gts = run_evaluate(config, model, trainer, pt_eval_ds)
            print(task_name)
            scores = pt_eval_ds.compute_metrics(all_gts, all_preds)
            print(scores)
            logger.info('{}\t{}'.format(task_name, json.dumps(scores)))

    if args.do_init_eval_ocl:
        logger.info('--init eval ocl--')
        for task_name, ocl_eval_ds in ocl_eval_dss.items():
            all_preds, all_gts = run_evaluate(config, model, trainer, ocl_eval_ds)
            print(task_name)
            scores = ocl_eval_ds.compute_metrics(all_gts, all_preds)
            print(scores)
            logger.info('{}\t{}'.format(task_name, json.dumps(scores)))

    if args.do_train:
        pt_train_dss = load_background_train_dss(config, tokenizer)
        memory = MultiTaskMemory(config, config.pt_tasks, datasets=[pt_train_dss[task] for task in config.pt_tasks],
                                 tokenizer=tokenizer, collator=data_collator)
        trainer = MyTrainer(
            model=model, args=training_args, train_dataset=None, eval_dataset=None,
            tokenizer=tokenizer, data_collator=data_collator, memory=memory, config=config
        )
        optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay,
                          betas=(training_args.adam_beta1, training_args.adam_beta2), eps=training_args.adam_epsilon)

        # load optimizer
        # optimizer_path = os.path.join(config.model_name, "optimizer.pt")
        # if os.path.exists(optimizer_path):
        #     logger.info('Loading optimizer')
        #     opt_states = torch.load(optimizer_path)
        #     optimizer.load_state_dict(opt_states)

        if not args.ocl_tasks:
            args.ocl_tasks = config.ocl_tasks

        for task_id, task in enumerate(args.ocl_tasks):
            print('Starting OCL task {} {}'.format(task_id, task))

            # evaluate on ocl task before training
            all_preds, all_gts = run_evaluate(config, model, trainer, ocl_eval_dss[task])
            scores = ocl_eval_dss[task].compute_metrics(all_gts, all_preds)
            logger.info('{}\t{}\tOCL-before\t{}'.format(task, json.dumps(scores), task_id))

            ocl_eval_func = get_ocl_eval_func(config, model, trainer, ocl_eval_dss[task], task, task_id)
            train_ds = ocl_train_dss[task]
            train_loader = DataLoader(train_ds, collate_fn=data_collator,
                                      batch_size=training_args.per_device_train_batch_size,
                                      shuffle=True)
            trainer.ocl_train_single_task_er(train_loader, optimizer, on_epoch_end=ocl_eval_func)

            logger.info('Task\t{}\t{}'.format(task_id, task))
            # evaluate on pt tasks
            for task_name, pt_eval_ds in pt_eval_dss.items():
                all_preds, all_gts = run_evaluate(config, model, trainer, pt_eval_ds)
                scores = pt_eval_ds.compute_metrics(all_gts, all_preds)
                logger.info('{}\t{}\tPT'.format(task_name, json.dumps(scores)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument("--do_init_eval_pt", action='store_true')
    parser.add_argument("--do_init_eval_ocl", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--ocl_tasks", nargs='*')

    args = parser.parse_args()
    run_pipeline(args)
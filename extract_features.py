from ocl_p3 import load_background_train_dss, load_background_eval_dss, load_ocl_dss, run_evaluate, run_evaluate_batch
from utils.config import load_configs
from trainer.my_trainer import MyTrainer
from trainer.utils import filter_params_by_regex
from trainer.utils import trim_batch, DataCollatorWithPaddingStr
from transformers import BartForConditionalGeneration, AutoTokenizer, TrainingArguments
from utils.config import merge_config_into_args, load_configs
from data_utils.p3 import P3Dataset, P3ConcatDataset, save_dataset
from data_utils.nli import Subset
import os
import logging
import re
import torch
import pickle
import argparse
from tqdm import tqdm
from utils.net_misc import disentangle_bart_embedding_weights
from torch.nn.functional import cross_entropy
from trainer.hooks import filter_modules_by_regex

from instance_ocl_p3 import load_bg_train_ds
from torch import nn

logger = logging.getLogger()

def collect_hooked_info_input_only(modules):
    xs, error_sigs = {}, {}
    for name, module in modules.items():
        if hasattr(module.weight, '__x__'):
            xs[name] = module.weight.__x__.detach().cpu()
            #error_sigs[name] = module.weight.__delta__.detach().cpu()
    return xs

def linear_forward_hook_mb(mod, activations, output):
    assert len(activations) == 1
    mod.weight.__x__ = activations[0].detach()

def hook_model(model, modules):
    handles = []
    for _, m in modules.items():
        handles.append(m.register_forward_hook(linear_forward_hook_mb))
    model.handles = handles


def run_feature_extraction_hooked(config, model, trainer: MyTrainer, eval_ds):
    eval_dataloader = trainer.get_eval_dataloader_raw(eval_ds)
    all_layer_inputs = {}

    linear_modules = filter_modules_by_regex(model, None, [nn.Linear])
    hook_model(model, linear_modules)

    model.eval()
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        with torch.no_grad():
            batch_ = trainer.clean_batch(batch)
            bs = batch_['input_ids'].size(0)

            batch_['input_ids'], batch_['attention_mask'], batch_['labels'] = batch_['input_ids'].cuda(), \
                                                                              batch_['attention_mask'].cuda(), \
                                                                              batch_['labels'].cuda()
            # get loss
            loss = trainer.compute_loss(model, batch_)

            layer_xs = collect_hooked_info_input_only(linear_modules)

            for k,v in layer_xs.items():
                #if k in param_names:
                if re.match(args.re_patt, k):
                    if k not in all_layer_inputs:
                        all_layer_inputs[k] = list(v.cpu().split(1))
                    else:
                        all_layer_inputs[k].extend(v.cpu().split(1))

    return all_layer_inputs

def run_feature_extraction_and_stat(config, model, trainer: MyTrainer, eval_ds):
    eval_dataloader = trainer.get_eval_dataloader_raw(eval_ds)
    all_preds, all_gts = [], []
    all_losses = []
    all_hiddens = []
    model.eval()
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        with torch.no_grad():
            input_ids, attn_masks = batch['input_ids'], batch['attention_mask']
            input_ids, attn_masks = trim_batch(input_ids, trainer.tokenizer.pad_token_id, attn_masks)
            input_ids, attn_masks = input_ids.cuda(), attn_masks.cuda()
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attn_masks,
                num_beams=config.num_beams,
                max_length=config.max_output_length,
                decoder_start_token_id=model.config.bos_token_id
            )

            for b_idx in range(len(input_ids)):
                pred_ = trainer.tokenizer.decode(outputs[b_idx])
                pred = trainer.tokenizer.decode(outputs[b_idx], skip_special_tokens=True)
                gt = batch['original_answers'][b_idx]
                all_preds.append(pred)
                all_gts.append(gt)

            batch_ = trainer.clean_batch(batch)

            batch_['input_ids'], batch_['attention_mask'], batch_['labels'] = batch_['input_ids'].cuda(), \
                                                                              batch_['attention_mask'].cuda(), \
                                                                              batch_['labels'].cuda()
            batch_['output_hidden_states'] = True
            # get loss
            _, outputs = trainer.compute_loss(model, batch_, return_outputs=True)
            logits = outputs.logits
            raw_loss = cross_entropy(logits.view(-1, logits.size(-1)), batch_['labels'].view(-1), reduction='none')
            raw_loss = raw_loss.view(logits.size(0), logits.size(1)) # [B,T]

            dec_hidden_states = outputs.decoder_hidden_states
            last_hidden_states = dec_hidden_states[-1]

            for b_idx in range(len(raw_loss)):
                b_len = (batch_['labels'][b_idx] != -100).sum()
                b_loss = raw_loss[b_idx,:b_len].cpu().numpy()
                all_losses.append(b_loss)
                all_hiddens.append(last_hidden_states[b_idx, :b_len].cpu().numpy())

    return all_preds, all_gts, all_losses, all_hiddens

def stat_bg_outputs(args):
    config = load_configs(*args.config_files, templates=args.templates)

    model = BartForConditionalGeneration.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    data_collator = DataCollatorWithPaddingStr(tokenizer)

    training_args = TrainingArguments(output_dir=config.output_dir)
    merge_config_into_args(config, training_args)

    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    trainer = MyTrainer(
        model=model, args=training_args, train_dataset=None, eval_dataset=None,
        tokenizer=tokenizer, data_collator=data_collator, memory=None, config=config
    )

    # pt_tasks = config.pt_tasks
    #
    # all_pt_dss = {}
    # for pt_task in pt_tasks:
    #     pt_ds = load_bg_train_ds(config, tokenizer, pt_task, max_example=config.max_bg_per_task)
    #     all_pt_dss[pt_task] = pt_ds

    if args.type == 'pt':
        pt_ds_path = os.path.join(config.output_dir, 'concat_pt_ds.csv')
        print('Extracting features for concat pt ds')
        concat_pt_ds = P3Dataset.from_csv(pt_ds_path, config, tokenizer)
        if args.final_layer_only:
            all_preds, all_gts, all_losses, all_hiddens = run_feature_extraction_and_stat(config, model, trainer, concat_pt_ds)
            save_records(config, {'all_preds': all_preds, 'all_gts': all_gts, 'all_losses': all_losses,
                                  'all_hiddens': all_hiddens}, 'pt')
        else:
            all_layer_inputs = run_feature_extraction_hooked(config, model, trainer, concat_pt_ds)
            save_records(config, {'all_layer_inputs': all_layer_inputs}, 'pt')
    elif args.type == 'ocl':
        ocl_ds_path = os.path.join(config.output_dir, 'ocl_error_ds.csv')
        print('Extracting features for concat ocl ds')
        ocl_ds = P3Dataset.from_csv(ocl_ds_path, config, tokenizer)
        if args.final_layer_only:
            all_preds, all_gts, all_losses, all_hiddens = run_feature_extraction_and_stat(config, model, trainer, ocl_ds)
            save_records(config, {'all_preds': all_preds, 'all_gts': all_gts, 'all_losses': all_losses,
                                  'all_hiddens': all_hiddens}, 'ocl_errors')
        else:
            all_layer_inputs = run_feature_extraction_hooked(config, model, trainer, ocl_ds)
            save_records(config, {'all_layer_inputs': all_layer_inputs}, 'ocl')


def save_records(config, records, name):
    save_dir = os.path.join(config.output_dir, 'features')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir,'{}.pkl'.format(name)),'wb') as wf:
        pickle.dump(records, wf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument("--ocl_task")
    parser.add_argument("--type")
    parser.add_argument("--final_layer_only", action='store_true')
    parser.add_argument("--re_patt")

    args = parser.parse_args()
    stat_bg_outputs(args)
import torch
import logging

logger = logging.getLogger()
import argparse

from utils.analysis_tools import *
from utils.net_misc import disentangle_bart_embedding_weights
from trainer.utils import filter_params_by_regex
from torch.optim import SGD
from tqdm import tqdm
import pickle
from torch import nn

import gc
from trainer.hooks import hook_model, filter_modules_by_regex

def save_prod_infos(config, prod_infos, exp_name):
    with open(os.path.join(config.output_dir,f'prod.{exp_name}.np.pkl'),'wb') as wf:
        pickle.dump(prod_infos, wf)

def save_grad_infos(config, prod_infos, offset, name):
    with open(os.path.join(config.output_dir,'grads-{}-{}.pkl'.format(name, offset)),'wb') as wf:
        pickle.dump(prod_infos, wf)

def get_configs_from_dirname(output_dir):
    items = output_dir.split('/')
    task = items[-1]
    exp = items[-2]
    model_type = items[-3][len('instance-p3-'):]
    return exp, task, model_type

def collect_hooked_info(modules):
    xs, error_sigs = {}, {}
    for name, module in modules.items():
        if hasattr(module.weight, '__x__'):
            xs[name] = module.weight.__x__.detach().cpu()
            error_sigs[name] = module.weight.__delta__.detach().cpu()
    return xs, error_sigs

def do_backward(model, batch, trainer):
    model.zero_grad()
    batch_ = trainer.clean_batch(batch)
    loss = trainer.training_step(model, batch_)
    return

def stat_grad_ocl(args):
    config, model, tokenizer, trainer, collator = initialize(args.config_files, args.templates)
    linear_modules = filter_modules_by_regex(model, None, [nn.Linear])
    hook_model(model, linear_modules)

    pt_ds_path = os.path.join(config.output_dir, 'concat_pt_ds.csv')
    concat_pt_ds = P3Dataset.from_csv(pt_ds_path, config, tokenizer)
    ocl_ds_path = os.path.join(config.output_dir, 'ocl_error_ds.csv')
    ocl_ds = P3Dataset.from_csv(ocl_ds_path, config, tokenizer)

    optim_params = model.parameters()
    training_args = trainer.args
    if config.optimizer_type == 'AdamW':
        optimizer = AdamW(optim_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay,
                          betas=(training_args.adam_beta1, training_args.adam_beta2),
                          eps=training_args.adam_epsilon)
    elif config.optimizer_type == 'SGD':
        optimizer = SGD(optim_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        print('Using SGD, configs: {}'.format(optimizer))
    else:
        raise NotImplementedError

    model_state, optim_state = save_model_state(model, optimizer)
    model.eval()

    all_before_ocl_xs, all_before_ocl_deltas = {}, {}
    all_after_ocl_xs, all_after_ocl_deltas = {}, {}
    save_idx = 0
    for ocl_idx in tqdm(range(len(ocl_ds))):
        # reset
        reset(model, optimizer, model_state, optim_state)

        # use old model to get grad
        ocl_example = ocl_ds[ocl_idx]
        ocl_batch = make_batch(ocl_example, collator)
        do_backward(model, ocl_batch, trainer)
        before_ocl_xs, before_ocl_deltas = collect_hooked_info(linear_modules)
        all_before_ocl_xs[ocl_idx], all_before_ocl_deltas[ocl_idx] = before_ocl_xs, before_ocl_deltas

        update_on_example(config, model, trainer, ocl_batch, optimizer)

        after_ocl_xs, after_ocl_deltas = collect_hooked_info(linear_modules)
        all_after_ocl_xs[ocl_idx], all_after_ocl_deltas[ocl_idx] = after_ocl_xs, after_ocl_deltas

        if (1 + ocl_idx) % args.part_size == 0 or ocl_idx == len(ocl_ds) - 1:

            logger.info('Saving grad infos to part {}'.format(ocl_idx))
            save_grad_infos(config, {'before_ocl_xs': all_before_ocl_xs,
                                     'before_ocl_deltas': all_before_ocl_deltas},
                            offset=save_idx, name='before_ocl')
            save_grad_infos(config, {'after_ocl_xs': all_after_ocl_xs,
                                     'after_ocl_deltas': all_after_ocl_deltas},
                            offset=save_idx, name='after_ocl')

            all_before_ocl_xs.clear()
            all_before_ocl_deltas.clear()
            all_after_ocl_xs.clear()
            all_after_ocl_deltas.clear()
            gc.collect()
            save_idx += 1

def stat_grad_pt(args):
    config, model, tokenizer, trainer, collator = initialize(args.config_files, args.templates)
    linear_modules = filter_modules_by_regex(model, None, [nn.Linear])
    hook_model(model, linear_modules)

    pt_ds_path = os.path.join(config.output_dir, 'concat_pt_ds.csv')
    concat_pt_ds = P3Dataset.from_csv(pt_ds_path, config, tokenizer)

    model.eval()

    all_pt_xs, all_pt_deltas = {}, {}
    for pt_idx in tqdm(range(len(concat_pt_ds))):
        # use old model to get grad
        pt_example = concat_pt_ds[pt_idx]
        ocl_batch = make_batch(pt_example, collator)
        do_backward(model, ocl_batch, trainer)
        before_ocl_xs, before_ocl_deltas = collect_hooked_info(linear_modules)
        all_pt_xs[pt_idx], all_pt_deltas[pt_idx] = before_ocl_xs, before_ocl_deltas


        if (1 + pt_idx) % args.part_size == 0:
            logger.info('Saving grad infos to part {}'.format(pt_idx))
            save_grad_infos(config, {'pt_xs': all_pt_xs,
                                     'pt_deltas': all_pt_deltas}, offset=pt_idx // args.part_size, name='pt')
            all_pt_xs.clear()
            all_pt_deltas.clear()
            gc.collect()
            #all_pt_xs, all_pt_deltas = {}, {}

def stat_grad_prods(args):
    config, model, tokenizer, trainer, collator = initialize(args.config_files, args.templates)
    _, frozen_old_model, _, _, _ = initialize(args.config_files, args.templates)

    #exp, task, model_type = get_configs_from_dirname(config.output_dir)
    #records = load_all_records(exp=exp, task=task, model_type=model_type)

    pt_ds_path = os.path.join(config.output_dir, 'concat_pt_ds.csv')
    concat_pt_ds = P3Dataset.from_csv(pt_ds_path, config, tokenizer)
    ocl_ds_path = os.path.join(config.output_dir, 'ocl_error_ds.csv')
    ocl_ds = P3Dataset.from_csv(ocl_ds_path, config, tokenizer)

    optim_params = model.parameters()
    if config.optim_module_regex:
        disentangle_bart_embedding_weights(model)
        optim_named_params = filter_params_by_regex(model, config.optim_module_regex)
        logger.info('Optimizing {}'.format([_ for _ in optim_named_params.keys()]))
        logger.info('Learning rate: {}'.format(trainer.args.learning_rate))
        optim_params = [_ for _ in optim_named_params.values()]

    optimizer = AdamW(optim_params, lr=trainer.args.learning_rate, weight_decay=trainer.args.weight_decay,
                      betas=(trainer.args.adam_beta1, trainer.args.adam_beta2), eps=trainer.args.adam_epsilon)

    model_state, optim_state = save_model_state(model, optimizer)

    model.eval()
    frozen_old_model.eval()

    prod_infos = {}

    for ocl_idx in tqdm(range(len(ocl_ds))):
        prod_infos[ocl_idx] = {}

        # reset
        reset(model, optimizer, model_state, optim_state)

        # use old model to get grad
        ocl_example = ocl_ds[ocl_idx]
        ocl_batch = make_batch(ocl_example, collator)
        ocl_before_grads = get_raw_gradients(config, model, ocl_batch, trainer, clip=True)
        update_on_example(config, model, trainer, ocl_batch, optimizer, eval_mode=True)
        ocl_after_grads = get_raw_gradients(config, model, ocl_batch, trainer, clip=True)

        param_diffs = get_param_differences(frozen_old_model.state_dict(), model.state_dict())

        for pt_idx in range(len(concat_pt_ds)):
            if pt_idx % 10 == 0: print(ocl_idx, pt_idx)
            pt_example = concat_pt_ds[pt_idx]
            pt_batch = make_batch(pt_example, collator)

            pt_before_grads = get_raw_gradients(config, frozen_old_model, pt_batch, trainer, clip=True)
            pt_after_grads = get_raw_gradients(config, model, pt_batch, trainer, clip=True)

            with torch.no_grad():
                before_inter_prod, before_inter_dist = get_prod_and_dist(ocl_before_grads, pt_before_grads)
                after_inter_prod, after_inter_dist = get_prod_and_dist(ocl_after_grads, pt_after_grads)
                ba_ocl_prod, ba_ocl_dist = get_prod_and_dist(ocl_before_grads, ocl_after_grads)
                ba_pt_prod, ba_pt_dist = get_prod_and_dist(pt_before_grads, pt_after_grads)

                before_chg_prod, before_chg_dist = get_prod_and_dist(param_diffs, pt_before_grads)
                after_chg_prod, after_chg_dist = get_prod_and_dist(param_diffs, pt_after_grads)

                before_chg_grad_prod, before_chg_grad_dist = get_prod_and_dist(param_diffs, ocl_before_grads)
                after_chg_grad_prod, after_chg_grad_dist = get_prod_and_dist(param_diffs, ocl_after_grads)

            del pt_before_grads
            del pt_after_grads

            prod_infos[ocl_idx][pt_idx] = {
                'before_inter_prod': before_inter_prod,
                'before_inter_dist': before_inter_dist,
                'after_inter_prod': after_inter_prod,
                'after_inter_dist': after_inter_dist,
                'ba_ocl_prod': ba_ocl_prod,
                'ba_ocl_dist': ba_ocl_dist,
                'ba_pt_prod': ba_pt_prod,
                'ba_pt_dist': ba_pt_dist,
                'before_chg_prod': before_chg_prod,
                'before_chg_dist': before_chg_dist,
                'after_chg_prod': after_chg_prod,
                'after_chg_dist': after_chg_dist,
                'before_chg_grad_prod': before_chg_grad_prod,
                'before_chg_grad_dist': before_chg_grad_dist,
                'after_chg_grad_prod': after_chg_grad_prod,
                'after_chg_grad_dist': after_chg_grad_dist
            }

        save_prod_infos(config, prod_infos, exp_name=args.exp_name)

def stat_grad_prods(args):
    config, model, tokenizer, trainer, collator = initialize(args.config_files, args.templates)
    _, frozen_old_model, _, _, _ = initialize(args.config_files, args.templates)

    #exp, task, model_type = get_configs_from_dirname(config.output_dir)
    #records = load_all_records(exp=exp, task=task, model_type=model_type)

    pt_ds_path = os.path.join(config.output_dir, 'concat_pt_ds.csv')
    concat_pt_ds = P3Dataset.from_csv(pt_ds_path, config, tokenizer)
    ocl_ds_path = os.path.join(config.output_dir, 'ocl_error_ds.csv')
    ocl_ds = P3Dataset.from_csv(ocl_ds_path, config, tokenizer)

    optim_params = model.parameters()
    if config.optim_module_regex:
        disentangle_bart_embedding_weights(model)
        optim_named_params = filter_params_by_regex(model, config.optim_module_regex)
        logger.info('Optimizing {}'.format([_ for _ in optim_named_params.keys()]))
        logger.info('Learning rate: {}'.format(trainer.args.learning_rate))
        optim_params = [_ for _ in optim_named_params.values()]

    optimizer = AdamW(optim_params, lr=trainer.args.learning_rate, weight_decay=trainer.args.weight_decay,
                      betas=(trainer.args.adam_beta1, trainer.args.adam_beta2), eps=trainer.args.adam_epsilon)

    model_state, optim_state = save_model_state(model, optimizer)

    model.eval()
    frozen_old_model.eval()

    prod_infos = {}

    for ocl_idx in tqdm(range(len(ocl_ds))):
        prod_infos[ocl_idx] = {}

        # reset
        reset(model, optimizer, model_state, optim_state)

        # use old model to get grad
        ocl_example = ocl_ds[ocl_idx]
        ocl_batch = make_batch(ocl_example, collator)
        print(ocl_batch['input_ids'].device)
        ocl_before_grads = get_raw_gradients(config, model, ocl_batch, trainer, clip=True)
        update_on_example(config, model, trainer, ocl_batch, optimizer, eval_mode=True)
        ocl_after_grads = get_raw_gradients(config, model, ocl_batch, trainer, clip=True)

        param_diffs = get_param_differences(frozen_old_model.state_dict(), model.state_dict())

        for pt_idx in range(len(concat_pt_ds)):
            if pt_idx % 10 == 0: print(ocl_idx, pt_idx)
            pt_example = concat_pt_ds[pt_idx]
            pt_batch = make_batch(pt_example, collator)

            print(pt_batch['input_ids'].device)
            pt_before_grads = get_raw_gradients(config, frozen_old_model, pt_batch, trainer, clip=True)
            pt_after_grads = get_raw_gradients(config, model, pt_batch, trainer, clip=True)
            print(pt_batch['input_ids'].device)

            with torch.no_grad():
                before_inter_prod, before_inter_dist = get_prod_and_dist(ocl_before_grads, pt_before_grads)
                after_inter_prod, after_inter_dist = get_prod_and_dist(ocl_after_grads, pt_after_grads)
                ba_ocl_prod, ba_ocl_dist = get_prod_and_dist(ocl_before_grads, ocl_after_grads)
                ba_pt_prod, ba_pt_dist = get_prod_and_dist(pt_before_grads, pt_after_grads)

                before_chg_prod, before_chg_dist = get_prod_and_dist(param_diffs, pt_before_grads)
                after_chg_prod, after_chg_dist = get_prod_and_dist(param_diffs, pt_after_grads)

                before_chg_grad_prod, before_chg_grad_dist = get_prod_and_dist(param_diffs, ocl_before_grads)
                after_chg_grad_prod, after_chg_grad_dist = get_prod_and_dist(param_diffs, ocl_after_grads)

            del pt_before_grads
            del pt_after_grads

            prod_infos[ocl_idx][pt_idx] = {
                'before_inter_prod': before_inter_prod,
                'before_inter_dist': before_inter_dist,
                'after_inter_prod': after_inter_prod,
                'after_inter_dist': after_inter_dist,
                'ba_ocl_prod': ba_ocl_prod,
                'ba_ocl_dist': ba_ocl_dist,
                'ba_pt_prod': ba_pt_prod,
                'ba_pt_dist': ba_pt_dist,
                'before_chg_prod': before_chg_prod,
                'before_chg_dist': before_chg_dist,
                'after_chg_prod': after_chg_prod,
                'after_chg_dist': after_chg_dist,
                'before_chg_grad_prod': before_chg_grad_prod,
                'before_chg_grad_dist': before_chg_grad_dist,
                'after_chg_grad_prod': after_chg_grad_prod,
                'after_chg_grad_dist': after_chg_grad_dist
            }

        save_prod_infos(config, prod_infos, exp_name=args.exp_name)

if __name__ == '__main__':
    #TASK = 'super_glue-cb'
    #configs = ['configs/p3/p3_default.yaml', 'configs/p3/instance-bart0-base-ocl/vanilla_bg100_large.yaml',
    #           'configs/p3/instance-bart0-base-ocl/steps.yaml', 'configs/p3/instance-bart0-base-ocl/greedy.yaml']
    #templates = [f"postfix=_lr1e-6_step30_greedy/{TASK}"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')
    parser.add_argument("--ocl_task")
    parser.add_argument("--type", choices=['pt','ocl','prod'])
    parser.add_argument("--part_size", type=int, default=100)
    parser.add_argument("--exp_name", default='new')

    args = parser.parse_args()
    if args.type == 'ocl':
        stat_grad_ocl(args)
    elif args.type == 'pt':
        stat_grad_pt(args)
    elif args.type == 'prod':
        stat_grad_prods(args)
    else:
        raise NotImplementedError(args.type)
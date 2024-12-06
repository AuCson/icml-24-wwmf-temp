from data_utils.p3 import P3Dataset, P3ConcatDataset, save_dataset, load_bg_train_ds
from utils.analysis_tools import initialize, make_batch
import argparse
from collections import OrderedDict
import torch
from torch.nn import functional as F
from trainer.utils import filter_modules_by_regex
from trainer.hooks import hook_model
from torch import nn
import numpy as np
from torch.utils.data import Subset
import os
import pickle

def collect_grads(model):
    grads = OrderedDict({n:p.grad for n,p in model.named_parameters()})
    vs = []
    for n, g in grads.items():
        if g is not None:
            vs.append(g.view(-1))
    vs = torch.cat(vs)
    return vs

def collect_hooked_info(modules):
    xs, error_sigs = {}, {}
    for name, module in modules.items():
        if hasattr(module.weight, '__x__'):
            xs[name] = module.weight.__x__.detach()
            error_sigs[name] = module.weight.__delta__.detach()
    return xs, error_sigs


def get_gradients(model, trainer, batch):
    model.zero_grad()
    batch = trainer._prepare_inputs(batch)
    batch_ = trainer.clean_batch(batch)
    loss = trainer.compute_loss(model, batch_)
    loss.backward()
    # grad_v = collect_grads(model)

    xs, error_sigs = collect_hooked_info(linear_modules)
    return xs, error_sigs

def get_grad_for_ds(model, trainer, collator, ds):
    all_xs, all_error_sigs = [], []
    for idx in range(len(ds)):
        example = ds[idx]
        batch = make_batch(example, collator)
        xs, sigs = get_gradients(model, trainer, batch)
        all_xs.append(xs)
        all_error_sigs.append(sigs)
    return all_xs, all_error_sigs

def get_prod(x1, sig1, x2, sig2):
    x1, sig1, x2, sig2 = x1.view(-1, x1.size(2)), sig1.view(-1, sig1.size(2)), x2.view(-1, x2.size(2)), sig2.view(-1,sig2.size(2))
    g1 = torch.matmul(x1.transpose(0, 1), sig1).view(-1)
    g2 = torch.matmul(x2.transpose(0, 1), sig2).view(-1)
    return (g1 * g2).sum()

def compute_dot_prod(xs1, sigs1, xs2, sigs2):
    s = 0
    for key in xs1:
        x1, sig1, x2, sig2 = xs1[key], sigs1[key], xs2[key], sigs2[key] # [1,T,H1] [1,T,H2]

        s += get_prod(x1, sig1, x2, sig2)
    return s

# def compute_dot_prod_1vn(xs1, sigs1, other_xs2, other_sigs2):
#     s = 0
#     for key in xs1:
#         key_s = 0
#         x1, sig1 = xs1[key], sigs1[key]
#         other_x2, other_sig2 = [x[key] for x in other_xs2], [x[key] for x in other_sigs2]
#         for x2, sig2 in zip(other_x2, other_sig2):
#             ts = get_prod(x1, sig1, x2, sig2)
#             key_s += ts


def compute_coreset_affinity(model, trainer, ds):
    all_xs, all_error_sigs = get_grad_for_ds(model, trainer, collator, ds) # [B,P]
    with torch.no_grad():
        prods = np.zeros((len(ds), len(ds)))
        for i in range(len(ds)):
            for j in range(len(ds)):
                prods[i,j] = compute_dot_prod(all_xs[i], all_error_sigs[i], all_xs[j], all_error_sigs[j])
    return prods

# def compute_coreset_affinity(model, trainer, ds):
#     all_grads = get_grad_for_ds(model, trainer, collator, ds) # [B,P]
#     with torch.no_grad():
#         mean_task_grad = all_grads.mean(0)
#
#         # batch sim
#         batch_sim = F.cosine_similarity(mean_task_grad.view(-1,1), all_grads)
#         N = all_grads.size(0)
#
#         # pairwise sim
#         all_grads_xxyy = torch.repeat_interleave(all_grads, N, 0)
#         all_grads_xyxy = all_grads.repeat(N, 1)
#
#         pair_sim = F.cosine_similarity(all_grads_xxyy, all_grads_xyxy)
#         sim_mat = pair_sim.view(N, N)
#
#         sim_mat.masked_fill_(torch.eye(N).bool(), 0.)
#
#         avg_sim = -1.0 / (N - 1) * sim_mat.sum(-1)
#
#         sum_score = avg_sim + batch_sim
#         return sum_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')
    parser.add_argument("--chunk_size", type=int, default=10)
    args = parser.parse_args()

    config, base_model, tokenizer, base_trainer, collator = initialize(args.config_files, args.templates)
    base_model = base_model.cuda()
    pt_tasks = config.pt_tasks

    linear_modules = filter_modules_by_regex(base_model, None, [nn.Linear])
    hook_model(base_model, linear_modules)

    coreset_scores = {}

    config.max_input_length = 256
    config.max_output_length = 64
    config.truncate_prefix = True

    for pt_task in pt_tasks:
        print(pt_task)
        coreset_scores[pt_task] = []
        pt_ds = load_bg_train_ds(config, tokenizer, pt_task, max_example=config.max_bg_per_task,
                                 offset=config.pt_ds_offset)

        for i in range(0, len(pt_ds), args.chunk_size):
            pt_ds_ss = Subset(pt_ds, [_ for _ in range(i, i+args.chunk_size)])
            cs_score = compute_coreset_affinity(base_model, base_trainer, pt_ds_ss)
            coreset_scores[pt_task].append(cs_score)

        with open(os.path.join(config.output_dir, 'coreset_stat.pkl'), 'wb') as wf:
            pickle.dump(coreset_scores, wf)



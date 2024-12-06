from utils.analysis_tools import *
from torch.optim import RMSprop, SGD
from utils.logit_predict_tools import update_on_example_eval_mode

from stat_logits import stat_logits_over_ds
from torch.utils.data import Dataset
import argparse

class SlicedDataset(Dataset):
    def __init__(self, ds, sz):
        super().__init__()
        self.ds = ds
        self.indices = [_ for _ in range(sz)]

    def __getitem__(self, idx):
        return self.ds[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def run_evaluate_batch(config, model, trainer, batch):
    batch_preds, batch_gts = [], []
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
        # pred = trainer.tokenizer.decode(outputs[b_idx], skip_special_tokens=True)
        gt = batch['original_answers'][b_idx]
        batch_preds.append(pred_)
        batch_gts.append(gt)
    return batch_preds, batch_gts


def fix_label_encoding(encoding, tokenizer):
    new_encoding = {}
    n = encoding['input_ids'].size(0)
    new_encoding['input_ids'] = torch.cat([torch.full(size=(n, 2), fill_value=tokenizer.bos_token_id),
                                           encoding['attention_mask']], 1)
    new_encoding['attention_mask'] = torch.cat([torch.full(size=(n, 2), fill_value=tokenizer.bos_token_id),
                                                1], 1)
    return new_encoding


def fix_batch(batch, tokenizer):
    # return batch

    new_batch = {k: v for k, v in batch.items()}
    n = new_batch['labels'].size(0)
    new_batch['labels'] = torch.cat([torch.full(size=(n, 2), fill_value=tokenizer.bos_token_id),
                                     new_batch['labels']], 1)
    return new_batch


def l2_dist(v, cand):
    s = (v - cand) ** 2
    d = torch.sqrt(s.sum(-1))
    return d


def find_matching_ts(pt_logits_change, ocl_logits_change, pt_labels, ocl_labels, topk=10):  # [T,V]
    rets = []
    for t_p in range(len(pt_labels)):
        pt_chg = pt_logits_change[t_p]
        pt_chg_topv, pt_chg_topi = pt_chg.topk(topk)  # [K]

        ocl_chg = ocl_logits_change[:ocl_labels.size(0), pt_chg_topi]  # [T, K]

        # alignment
        dist = l2_dist(pt_chg_topv, ocl_chg)

        min_v, min_t = dist.min(-1)
        rets.append(min_t.item())

        #print(dist, pt_chg)
    return rets


def find_matching_ts_masked(pt_logits_change, ocl_logits_change, chg_mask, pt_logits, pt_labels, ocl_labels,
                            topk=20):  # [T,V]
    rets = []
    for t_p in range(len(pt_labels)):
        pt_chg = pt_logits_change[t_p]
        pt_chg_mask = chg_mask[t_p]

        # special treatment for </s>
        pt_chg[0] = 0

        # pt_chg_abs = pt_chg.abs()
        # pt_chg_abs_topv, pt_chg_topi = pt_chg_abs.topk(topk) # [K]
        # pt_chg_topv = pt_chg[pt_chg_abs_topi]

        pt_logits_topv, pt_chg_topi = pt_logits[t_p].topk(topk)
        pt_chg_topv = pt_chg[pt_chg_topi]

        ocl_logits_change[:, 0] = 0
        ocl_chg = ocl_logits_change[:ocl_labels.size(0), pt_chg_topi]  # [T, K]

        mask = pt_chg_mask[pt_chg_topi]

        # alignment
        #print(ocl_chg.size())
        dist = l2_dist(pt_chg_topv[mask], ocl_chg[:, mask])

        min_v, min_t = dist.min(-1)
        rets.append(min_t.item())

        #print(dist, pt_chg, mask)
        # print(ocl_chg)
    return rets


def unpack_logits(logits_tup, vocab_size):
    logits, idxs = logits_tup  # [T, Vs], [T, Vs]
    if not torch.is_tensor(logits):
        logits = torch.from_numpy(logits)
        idxs = torch.from_numpy(idxs)

    logits_full = torch.zeros(logits.size(0), vocab_size)
    logits_full.scatter_(1, idxs, logits)  # [T,V]
    mask = torch.zeros_like(logits_full).bool()
    mask.scatter_(1, idxs, 1)
    return logits_full, mask


def get_logits_change_ss(before_logits_tup, after_logits_tup, vocab_size):
    before_logits_full, before_logits_mask = unpack_logits(before_logits_tup, vocab_size)
    after_logits_full, after_logits_mask = unpack_logits(after_logits_tup, vocab_size)

    both_mask = before_logits_mask & after_logits_mask
    #print(both_mask.sum(), both_mask.size())

    logits_change = after_logits_full - before_logits_full
    return logits_change, both_mask, before_logits_full, after_logits_full


def get_before_after_logits_and_preds(config, ocl_error_idx, ocl_error_ds, pt_ds, model, trainer, optimizer,
                                      model_state, optim_state, tokenizer, collator):
    ocl_example = ocl_error_ds[ocl_error_idx]
    ocl_batch = make_batch(ocl_example, collator)
    reset(model, optimizer, model_state, optim_state)

    with torch.no_grad():
        model.eval()
        before_ocl_preds, before_ocl_gts = run_evaluate_batch(config, model, trainer, ocl_batch)
        before_ocl_logits = get_logit_batch(config, model, trainer, ocl_batch)
    print('Before pt logits')
    before_pt_logits, before_pt_labels = stat_logits_over_ds(config, pt_ds, model, trainer)

    update_on_example_eval_mode(config, model, trainer, ocl_batch, optimizer)

    print('After pt logits')
    after_pt_logits, after_pt_labels = stat_logits_over_ds(config, pt_ds, model, trainer)

    after_ocl_preds, after_ocl_gts = run_evaluate_batch(config, model, trainer, ocl_batch)
    after_ocl_logits = get_logit_batch(config, model, trainer, ocl_batch)

    return {
        'before_ocl_logits': before_ocl_logits,
        'before_ocl_labels': before_ocl_gts,
        'after_ocl_logits': after_ocl_logits,
        'after_ocl_labels': after_ocl_gts,
        'before_pt_logits': before_pt_logits,
        'before_pt_labels': before_pt_labels,
        'after_pt_logits': after_pt_logits,
        'after_pt_labels': after_pt_labels,
        'before_ocl_preds': before_ocl_preds,
        'after_ocl_preds': after_ocl_preds
    }


def main(args):
    TASK = args.ocl_task
    configs = args.config_files
    templates = args.templates
    config, model, tokenizer, trainer, collator = initialize(configs, templates)
    optim_params = model.parameters()
    optimizer = SGD(optim_params, lr=trainer.args.learning_rate)
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    optim_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state_dict().items()}
    optim_params = model.parameters()

    exp_name = config.output_dir.split('/')[2]
    print('Exp name: {}'.format(exp_name))
    #exp_name = 'vanilla_bg100_lr1e-6_step30_greedy_eval_fixbos'

    #pt_logits_file = os.path.join(config.output_dir, 'concat_pt_logits_eval.pkl')

    # ocl_update_logits_file = f'runs/instance-p3-bart0-large/vanilla_bg100_lr1e-6_step30_greedy/{TASK}/ocl_error_ds_change_v2_logit_change_eval.pkl'

    #ocl_update_logits_file = os.path.join(config.output_dir, 'ocl_error_ds_change_v2_logit_change_eval.pkl')

    #with open(pt_logits_file, 'rb') as f:
    #    all_pt_logits = pickle.load(f)

    #with open(ocl_update_logits_file, 'rb') as f:
    #    all_ocl_update_logits = pickle.load(f)

    pt_ds_path = os.path.join(config.output_dir, 'concat_pt_ds.csv')
    concat_pt_ds = P3Dataset.from_csv(pt_ds_path, config, tokenizer)

    ocl_ds_path = os.path.join(config.output_dir, 'ocl_error_ds.csv')
    ocl_ds = P3Dataset.from_csv(ocl_ds_path, config, tokenizer)

    records = load_all_records(exp=exp_name, task=TASK, model_type='bart0-large')
    all_ocl_idxs = [_ for _ in records['ocl_obj'].keys()]

    all_matching_infos = {}
    for ocl_error_idx in range(len(all_ocl_idxs)):
        print('OCL error idx', ocl_error_idx)
        all_matching_infos[ocl_error_idx] = {}

        ba_logits_preds = get_before_after_logits_and_preds(config, ocl_error_idx, ocl_ds, concat_pt_ds, model, trainer,
                                                            optimizer, model_state, optim_state, tokenizer, collator)

        for pt_idx in range(len(concat_pt_ds)):
            change, chg_mask, before_logits_full, after_logits_full = get_logits_change_ss(
                ba_logits_preds['before_pt_logits'][pt_idx], ba_logits_preds['after_pt_logits'][pt_idx],
                vocab_size=len(tokenizer))

            pt_labels = ba_logits_preds['before_pt_labels'][pt_idx]

            ocl_example = ocl_ds[ocl_error_idx]
            ocl_batch = make_batch(ocl_example, collator)
            ocl_labels = ocl_batch['labels'][0]

            ocl_change = ba_logits_preds['after_ocl_logits'] - ba_logits_preds['before_ocl_logits']
            ocl_change = ocl_change[0]

            matching = find_matching_ts_masked(change.cpu(), ocl_change.cpu(), chg_mask.cpu(), before_logits_full, pt_labels,
                                               ocl_labels[ocl_labels != 1], topk=10)
            all_matching_infos[ocl_error_idx][pt_idx] = matching

        with open(os.path.join(config.output_dir, 'matching.pkl'), 'wb') as wf:
            print('Saving...')
            pickle.dump(all_matching_infos, wf)
            print('Saved')

def main_debug():
    TASK = 'super_glue-copa'
    configs = ['configs/p3/p3_default.yaml', 'configs/p3/instance-bart0-base-ocl/vanilla_bg100_large.yaml',
               'configs/p3/instance-bart0-base-ocl/steps.yaml', 'configs/p3/instance-bart0-base-ocl/greedy.yaml',
               'configs/p3/instance-bart0-base-ocl/sgd.yaml', 'configs/p3/instance-bart0-base-ocl/lr1e-2.yaml']
    # templates = [f"postfix=_lr1e-6_step30_sgd_greedy_eval_fixbos-lr1e-2/{TASK}"]
    templates = [f"postfix=_lr1e-6_step50_sgd_greedy_eval_fixbos-lr1e-2/{TASK}"]
    config, model, tokenizer, trainer, collator = initialize(configs, templates)
    optim_params = model.parameters()
    optimizer = SGD(optim_params, lr=trainer.args.learning_rate)
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    optim_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state_dict().items()}

    optim_params = model.parameters()

    # pt_logits_file = f'runs/instance-p3-bart0-large/vanilla_bg100_lr1e-6_step30_greedy/{TASK}/concat_pt_logits_eval.pkl'

    exp_name = 'vanilla_bg100_lr1e-6_step30_sgd_greedy_eval_fixbos-lr1e-2'

    # exp_name = 'vanilla_bg100_lr1e-6_step30_greedy_eval_fixbos'

    pt_logits_file = f'runs/instance-p3-bart0-large/vanilla_bg100_lr1e-6_step30_sgd_greedy_eval_fixbos-lr1e-2/{TASK}/concat_pt_logits_eval.pkl'

    # ocl_update_logits_file = f'runs/instance-p3-bart0-large/vanilla_bg100_lr1e-6_step30_greedy/{TASK}/ocl_error_ds_change_v2_logit_change_eval.pkl'

    ocl_update_logits_file = f'runs/instance-p3-bart0-large/vanilla_bg100_lr1e-6_step30_sgd_greedy_eval_fixbos-lr1e-2/{TASK}/ocl_error_ds_change_v2_logit_change_eval.pkl'

    with open(pt_logits_file, 'rb') as f:
        all_pt_logits = pickle.load(f)

    with open(ocl_update_logits_file, 'rb') as f:
        all_ocl_update_logits = pickle.load(f)

    pt_ds_path = os.path.join(config.output_dir, 'concat_pt_ds.csv')
    concat_pt_ds = P3Dataset.from_csv(pt_ds_path, config, tokenizer)
    pt_ds = SlicedDataset(concat_pt_ds, 10)

    ocl_ds_path = os.path.join(config.output_dir, 'ocl_error_ds.csv')
    ocl_ds = P3Dataset.from_csv(ocl_ds_path, config, tokenizer)

    records = load_all_records(exp=exp_name, task=TASK, model_type='bart0-large')
    all_ocl_idxs = [_ for _ in records['ocl_obj'].keys()]

    ocl_error_idx = 0
    pt_idx = 4
    ba_logits_preds = get_before_after_logits_and_preds(config, ocl_error_idx, ocl_ds, pt_ds, model, trainer,
                                                        optimizer, model_state, optim_state, tokenizer, collator)
    change, chg_mask, before_logits_full, after_logits_full = get_logits_change_ss(
        ba_logits_preds['before_pt_logits'][pt_idx], ba_logits_preds['after_pt_logits'][pt_idx], vocab_size=len(tokenizer))

    pt_labels = ba_logits_preds['before_pt_labels'][pt_idx]

    ocl_example = ocl_ds[ocl_error_idx]
    ocl_batch = make_batch(ocl_example, collator)
    ocl_labels = ocl_batch['labels'][0]

    ocl_change = ba_logits_preds['after_ocl_logits'] - ba_logits_preds['before_ocl_logits']
    ocl_change = ocl_change[0]

    matching = find_matching_ts_masked(change.cpu(), ocl_change.cpu(), chg_mask.cpu(), before_logits_full, pt_labels, ocl_labels[ocl_labels!=1], topk=10)
    print(matching)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')
    parser.add_argument("--ocl_task")

    args = parser.parse_args()
    main(args)
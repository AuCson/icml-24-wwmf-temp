from utils.analysis_tools import *
import argparse
from trainer.my_trainer import MyTrainer
from transformers import TrainingArguments
from tqdm import tqdm
from data_utils.p3 import P3Dataset
from torch.optim import SGD
from torch.nn.functional import softmax
from trainer.utils import filter_params_by_regex
from utils.net_misc import disentangle_bart_embedding_weights

def stat_logits_over_ds(config, ds, model, trainer, return_probs=False, topk=100):
    dataloader = trainer.get_eval_dataloader_raw(ds)
    all_logits, all_labels = [], []
    all_probs = []
    model.eval()
    for batch in tqdm(dataloader, total=len(dataloader)):
        logits = get_logit_batch(config, model, trainer, batch) # [B,T,V]
        batch_size = logits.size(0)
        for b in range(batch_size):
            b_logits = logits[b]

            keep_mask = batch['labels'][b].ne(-100)

            b_logits = b_logits[keep_mask,:]
            b_labels = batch['labels'][b][keep_mask]

            topv, topi = b_logits.topk(topk)
            prob = softmax(b_logits, -1)
            topv_prob, _ = prob.topk(topk)

            all_logits.append([topv.cpu().detach().numpy(), topi.cpu().detach().numpy()])
            all_labels.append(b_labels.detach().numpy())
            all_probs.append(topv_prob.cpu().detach().numpy())
    if return_probs:
        return all_logits, all_labels, all_probs
    else:
        return all_logits, all_labels

def to_np(items):
    return [x.cpu().detach().numpy() for x in items]

def stat_logit_changes(config, ds, model, optimizer, trainer):
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    optim_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state_dict().items()}
    dataloader = trainer.get_eval_dataloader_raw(ds, batch_size=1)
    all_before_logits, all_after_logits, all_logits_change, all_labels = [], [], [], []
    K = 1000

    for batch in tqdm(dataloader, total=len(dataloader)):

        batch_size = batch['labels'].size(0)
        assert batch_size == 1

        model.eval()
        before_logits = get_logit_batch(config, model, trainer, batch) # [B,T,V]

        trainer.nstep_model_update(model, batch, optimizer, n_step=config.ocl_steps, eval_mode=True)

        model.eval()
        after_logits = get_logit_batch(config, model, trainer, batch) # [B,T,V]
        logit_change = after_logits - before_logits

        keep_mask = batch['labels'][0].ne(-100)

        all_before_logits.append(to_np(before_logits[0][keep_mask].topk(K)))
        all_after_logits.append(to_np(after_logits[0][keep_mask].topk(K)))
        all_logits_change.append(logit_change[0][keep_mask].cpu().detach().numpy())
        #logits_decrease.append(to_np(logit_change[0][keep_mask].topk(K, largest=False)))

        labels = batch['labels'][0][keep_mask]
        all_labels.append(labels.detach().numpy())

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)
    return all_before_logits, all_after_logits, all_logits_change, all_labels

def stat_logit_pt_after(config, ocl_ds, pt_ds, model, optimizer, trainer, topk):
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    optim_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state_dict().items()}
    dataloader = trainer.get_eval_dataloader_raw(ocl_ds, batch_size=1)
    all_before_logits, all_after_logits, all_logits_change, all_labels = [], [], [], []
    K = 1000

    res = []
    for ocl_error_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        batch_size = batch['labels'].size(0)
        assert batch_size == 1

        model.eval()

        trainer.nstep_model_update(model, batch, optimizer, n_step=config.ocl_steps, eval_mode=True)

        model.eval()

        after_pt_logits = stat_logits_over_ds(config, pt_ds, model, trainer, return_probs=True, topk=2)
        res.append({'labels': after_pt_logits[1], 'logits': after_pt_logits[0], 'probs': after_pt_logits[2]})

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)

    return res

def save_logit_infos(config, logits, labels, name):
    with open(os.path.join(config.output_dir, f'{name}_logits_eval.pkl'),'wb') as wf:
        pickle.dump({
            'labels': labels,
            'logits': logits
        }, wf)

def save_logit_change_infos(config, before_logits, after_logits, logits_change, labels, name):
    with open(os.path.join(config.output_dir, f'{name}_logit_change_eval.pkl'), 'wb') as wf:
        pickle.dump({
            'labels': labels,
            'before_logits': before_logits,
            'after_logits': after_logits,
            'logits_change': logits_change,
        }, wf)


def main(args):
    config, model, tokenizer, trainer, collator = initialize(args.config_files, args.templates)
    training_args = TrainingArguments(output_dir=config.output_dir)

    merge_config_into_args(config, training_args)
    trainer = MyTrainer(
        model=model, args=training_args, train_dataset=None, eval_dataset=None,
        tokenizer=tokenizer, data_collator=collator, memory=None, config=config
    )

    pt_ds_path = os.path.join(config.output_dir, 'concat_pt_ds.csv')
    if os.path.isfile(pt_ds_path):
        concat_pt_ds = P3Dataset.from_csv(pt_ds_path, config, tokenizer)
    else:
        concat_pt_ds = 0
    ocl_error_ds_path = os.path.join(config.output_dir, 'ocl_error_ds.csv')
    ocl_error_ds = P3Dataset.from_csv(ocl_error_ds_path, config, tokenizer)
    if args.update or args.update_pt:
        print('Stat logits of updated models, update steps={}'.format(config.ocl_steps))
        #optim_params = model.parameters()

        optim_params = [x for x in model.parameters() if x.requires_grad]
        if config.optim_module_regex:
            disentangle_bart_embedding_weights(model)
            optim_named_params = filter_params_by_regex(model, config.optim_module_regex)
            logger.info('Optimizing {}'.format([_ for _ in optim_named_params.keys()]))
            logger.info('Learning rate: {}'.format(training_args.learning_rate))
            optim_params = [_ for _ in optim_named_params.values()]

        #optimizer = AdamW(optim_params, lr=trainer.args.learning_rate, weight_decay=trainer.args.weight_decay,
        #                  betas=(trainer.args.adam_beta1, trainer.args.adam_beta2), eps=trainer.args.adam_epsilon)

        if config.optimizer_type == 'AdamW':
            optimizer = AdamW(optim_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay,
                              betas=(training_args.adam_beta1, training_args.adam_beta2),
                              eps=training_args.adam_epsilon)
        elif config.optimizer_type == 'SGD':
            optimizer = SGD(optim_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
            print('Using SGD, configs: {}'.format(optimizer))
        else:
            raise NotImplementedError

        if args.update:
            before_logits, after_logits, logits_change, all_labels = \
                stat_logit_changes(config, ocl_error_ds, model, optimizer, trainer)
            save_logit_change_infos(config, before_logits, after_logits, logits_change, all_labels,f'ocl_error_ds_change_v2{args.postfix}')
        if args.update_pt:
            res = stat_logit_pt_after(config, ocl_error_ds, concat_pt_ds, model, optimizer, trainer, topk=2)
            with open(os.path.join(config.output_dir, f'pt_logits_update_eval.topk2.fix.pkl'), 'wb') as wf:
                pickle.dump(res, wf)
    else:
        pt_logits, pt_labels = stat_logits_over_ds(config, concat_pt_ds, model, trainer)
        save_logit_infos(config, pt_logits, pt_labels, 'concat_pt')

        ocl_error_logits, ocl_error_labels = stat_logits_over_ds(config, ocl_error_ds, model, trainer)
        save_logit_infos(config, ocl_error_logits, ocl_error_labels, 'ocl_error_ds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')
    parser.add_argument("--ocl_task")
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--update_pt', action='store_true')
    parser.add_argument("--postfix", default="")

    args = parser.parse_args()

    main(args)
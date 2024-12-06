from ocl_p3 import load_background_train_dss, load_background_eval_dss, load_ocl_dss
from trainer.my_trainer import MyTrainer
from trainer.utils import filter_params_by_regex
from trainer.utils import trim_batch, DataCollatorWithPaddingStr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from utils.config import merge_config_into_args, load_configs
from data_utils.p3 import P3Dataset, P3ConcatDataset, save_dataset, load_bg_train_ds, load_ocl_alt_ans_ds, load_bg_test_ds
from data_utils.nli import Subset
import os
import logging
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim import SGD
import torch
import pickle
import argparse
from tqdm import tqdm
from utils.net_misc import disentangle_bart_embedding_weights
from utils.analysis_tools import stat_loss_on_ds, stat_scores_on_ds, iscores2errors, run_evaluate_batch, \
    stat_replayed_vs_others_scores, create_past_model, load_peft_model, create_extended_dataset
import random
import gc
import numpy as np

logger = logging.getLogger('main')

def stat_scores_all_pt_tasks(config, model, trainer, all_pt_dss):
    task2iscores = {}
    for task, ds in all_pt_dss.items():
        scores = stat_scores_on_ds(config, model, trainer, ds)
        task2iscores[task] = scores
    return task2iscores

def evaluate_on_dev_set(args, config, model, trainer, tokenizer, exp_name=''):
    pt_tasks = config.pt_tasks
    if config.cached_pt_test_ds is not None:
        logger.info('Using cached pt ds at {}'.format(config.cached_pt_test_ds))
        concat_pt_test_ds = P3Dataset.from_csv(config.cached_pt_test_ds, config, tokenizer)
    else:
        all_pt_test_dss = {}
        for pt_task in pt_tasks:
            pt_ds = load_bg_test_ds(config, tokenizer, pt_task, max_example=config.max_bg_test_per_task)
            all_pt_test_dss[pt_task] = pt_ds
        concat_pt_test_ds = P3ConcatDataset([v for k,v in all_pt_test_dss.items()])

    before_pt_iscores, before_pt_preds = stat_scores_on_ds(config, model, trainer, concat_pt_test_ds, return_preds=True)
    # before_pt_losses = stat_loss_on_ds(config, model, trainer, concat_pt_ds)
    before_pt_errors = iscores2errors(before_pt_iscores)
    mean_em, mean_f1 = np.mean([x[0] for x in before_pt_iscores]), np.mean([x[1] for x in before_pt_iscores])
    logger.info("PT dev set performance: {}, {}".format(mean_em, mean_f1))
    with open(os.path.join(config.output_dir, '{}_pt_test_preds_log.pkl'.format(exp_name)), 'wb') as wf:
        pickle.dump({'preds': before_pt_preds, 'scores': before_pt_iscores, 'errors': before_pt_errors}, wf)


def run_pipeline_stat_errors(args):
    config = load_configs(*args.config_files, templates=args.templates)

    print(config.output_dir)

    if config.is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.padding_side = 'left'

    if config.peft == 'lora':
        model = load_peft_model(config=config, base_model=model)

    if config.fix_decoder_start_token_id: # bart
        print('Fixing decoder start token id')
        model.config.decoder_start_token_id = 0

    if config.fix_bos_token_id:
        print('Fixing bos token id')
        model.config.bos_token_id = 2

    if args.load_base_ckpt:
        state_dict = torch.load(args.load_base_ckpt)
        #state_dict = {k[len('model.'):]: v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)

    data_collator = DataCollatorWithPaddingStr(tokenizer)

    training_args = TrainingArguments(output_dir=config.output_dir)
    merge_config_into_args(config, training_args)

    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    config.save(config.output_dir, 'config.json')

    trainer = MyTrainer(
        model=model, args=training_args, train_dataset=None, eval_dataset=None,
        tokenizer=tokenizer, data_collator=data_collator, memory=None, config=config,
        past_model_creator=create_past_model

    )

    logger.info('The seed is {}'.format(training_args.seed))

    pt_tasks = config.pt_tasks

    all_pt_dss = {}

    aff_log = {}
    aff_score_log = {}
    aff_loss_log = {}
    aff_cl_log = {}
    ocl_log = {}
    after_ocl_preds_log = {}
    after_all_pt_preds_log = {}

    if config.cached_pt_ds is not None:
        logger.info('Using cached pt ds at {}'.format(config.cached_pt_ds))
        concat_pt_ds = P3Dataset.from_csv(config.cached_pt_ds, config, tokenizer)
    else:
        for pt_task in pt_tasks:
            pt_ds = load_bg_train_ds(config, tokenizer, pt_task, max_example=config.max_bg_per_task, offset=config.pt_ds_offset)
            all_pt_dss[pt_task] = pt_ds
        concat_pt_ds = P3ConcatDataset([v for k,v in all_pt_dss.items()])

    if config.cached_ocl_error_ds is not None:
        print('Using cached ocl_error_ds at {}'.format(config.cached_ocl_error_ds))
        ocl_train_ds = ocl_error_ds = P3Dataset.from_csv(config.cached_ocl_error_ds, config, tokenizer)
        ocl_iscores = None
        ocl_errors_idxs = [_ for _ in range(len(ocl_error_ds))]
    else:
        if config.use_mmlu:
            ocl_train_ds = P3Dataset.from_mmlu(config.ocl_tasks, 'val', config, tokenizer, skip_encoding=False)
        elif config.use_bbh:
            ocl_train_ds = P3Dataset.from_bbh(config.ocl_tasks, 'val', config, tokenizer, skip_encoding=False)
        elif config.use_alt_p3:
            ocl_train_ds = load_ocl_alt_ans_ds(config, args.ocl_task, tokenizer)
        else:
            ocl_train_dss, _, _ = load_ocl_dss(config, tokenizer)
            ocl_train_ds = ocl_train_dss[args.ocl_task]

    if config.cached_ocl_error_ds is None or args.load_base_ckpt:
        ocl_iscores, (ocl_base_preds, ocl_base_gts) = stat_scores_on_ds(config, model, trainer, ocl_train_ds, return_preds=True)
        ocl_errors_idxs = iscores2errors(ocl_iscores)
        if config.use_bbh:
            print('Subsampling for bbh')
            ocl_errors_idxs = sorted(random.sample(ocl_errors_idxs, 100))

        ocl_error_ds = Subset(ocl_train_ds, ocl_errors_idxs)

        ocl_base_score = ocl_train_ds.group_score_by_task([x[0] for x in ocl_iscores])
        logger.info(ocl_base_score)
        logger.info(np.mean([x[0] for x in ocl_iscores][:args.max_step]))

        with open(os.path.join(config.output_dir, 'base_ocl_errors.pkl'),'wb') as wf:
            pickle.dump({'ocl_iscores': ocl_iscores, 'ocl_error_idxs': ocl_errors_idxs, 'ocl_base_preds': ocl_base_preds,
                         'ocl_base_gts': ocl_base_gts, 'ocl_base_score': ocl_base_score}, wf)
        if args.eval_ocl_only:
            exit(0)

    if args.skip_before_eval:
        before_pt_iscores, before_pt_errors = None, None
    else:
        if args.do_test_eval:
            evaluate_on_dev_set(args, config, model, trainer, tokenizer, 'start')

        before_pt_iscores, before_pt_preds = stat_scores_on_ds(config, model, trainer, concat_pt_ds, return_preds=True)
        #before_pt_losses = stat_loss_on_ds(config, model, trainer, concat_pt_ds)
        before_pt_errors = iscores2errors(before_pt_iscores)
        with open(os.path.join(config.output_dir, 'before_pt_preds_log.pkl'),'wb') as wf:
            pickle.dump(before_pt_preds, wf)

        if args.only_before_eval:
            exit(0)

    aff_log['before'] = before_pt_errors
    aff_score_log['before'] = before_pt_iscores
    #aff_loss_log['before'] = before_pt_losses

    if config.extend_ocl_ds:
        ocl_error_ds_, ocl_error_idxs_ = ocl_error_ds, ocl_errors_idxs
        ocl_error_ds = create_extended_dataset(ocl_error_ds, config.extend_multiplier)
        ocl_errors_idxs = [_ for _ in range(len(ocl_error_ds))]

    ocl_error_train_loader = DataLoader(ocl_error_ds, collate_fn=data_collator,batch_size=1,shuffle=False)

    optim_params = [x for x in model.parameters() if x.requires_grad]
    if config.optim_module_regex:
        disentangle_bart_embedding_weights(model)
        optim_named_params = filter_params_by_regex(model, config.optim_module_regex)
        logger.info('Optimizing {}'.format([_ for _ in optim_named_params.keys()]))
        logger.info('Learning rate: {}'.format(training_args.learning_rate))
        optim_params = [_ for _ in optim_named_params.values()]

    pred_forget = None
    if config.pred_forget_file:
        with open(config.pred_forget_file,'rb') as f:
            pred_forget = pickle.load(f)

    if config.optimizer_type == 'AdamW':
        optimizer = AdamW(optim_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay,
                    betas=(training_args.adam_beta1, training_args.adam_beta2), eps=training_args.adam_epsilon)
    elif config.optimizer_type == 'SGD':
        optimizer = SGD(optim_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        print('Using SGD, configs: {}'.format(optimizer))
    else:
        raise NotImplementedError

    save_dataset(ocl_error_ds, os.path.join(config.output_dir, 'ocl_error_ds.csv'))
    save_dataset(concat_pt_ds, os.path.join(config.output_dir, 'concat_pt_ds.csv'))

    if config.seperate_replay_optimizer:
        if config.optimizer_type == 'AdamW':
            replay_optimizer = AdamW(optim_params, lr=config.replay_optimizer_lr, weight_decay=training_args.weight_decay,
                              betas=(training_args.adam_beta1, training_args.adam_beta2),
                              eps=training_args.adam_epsilon)
        elif config.optimizer_type == 'SGD':
            replay_optimizer = SGD(optim_params, lr=config.replay_optimizer_lr, weight_decay=training_args.weight_decay)
            print('Using SGD, configs: {}'.format(optimizer))
    else:
        replay_optimizer = None

    aff_cl_log['meta'] = {'ocl_error_idxs': ocl_errors_idxs}

    for step, (ocl_idx, ocl_batch) in tqdm(enumerate(zip(ocl_errors_idxs, ocl_error_train_loader)),total=len(ocl_errors_idxs)):
        model_state = {k: v.clone() for k, v in model.state_dict().items()}
        optim_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state_dict().items()}

        if not config.in_context_learning:
            if config.cl_method == "vanilla":
                replayed_idxs = trainer.nstep_model_update(model, ocl_batch, optimizer, n_step=config.ocl_steps, replay_optimizer=replay_optimizer,
                                           eval_mode=config.use_eval_mode)
            elif config.cl_method == "er":
                replayed_idxs = trainer.nstep_model_update(model, ocl_batch, optimizer, n_step=config.ocl_steps, do_replay=True, pt_ds=concat_pt_ds,
                                           score_func=concat_pt_ds.get_score_func(), replay_optimizer=replay_optimizer,eval_mode=config.use_eval_mode)
            elif config.cl_method == "mir":
                replayed_idxs = trainer.nstep_model_update(model, ocl_batch, optimizer, n_step=config.ocl_steps, do_replay=True, do_retrieve=True,
                                           pt_ds=concat_pt_ds, score_func=concat_pt_ds.get_score_func(), replay_optimizer=replay_optimizer,eval_mode=config.use_eval_mode,
                                           )
            elif config.cl_method == "mir_pred":
                ocl_error_idx = step
                replayed_idxs = trainer.nstep_model_update(model, ocl_batch, optimizer, n_step=config.ocl_steps, do_replay=True, do_retrieve=False,
                                           pt_ds=concat_pt_ds, score_func=concat_pt_ds.get_score_func(), replay_optimizer=replay_optimizer,
                                           eval_mode=config.use_eval_mode, pred_forgets=pred_forget[ocl_error_idx], use_pred_forget=True)
            else:
                raise NotImplementedError

            if args.gc:
                torch.cuda.empty_cache()
                gc.collect()

        # see whether error is fixed

        eval_pt_ds = concat_pt_ds
        if config.in_context_learning:
            eval_pt_ds = P3Dataset.from_concat_example(config, tokenizer, in_context_input=ocl_batch['original_input'][0],
                                                       in_context_ans=ocl_batch['original_answers'][0], raw_ds=concat_pt_ds)

        if not args.no_intermed_eval and step % args.interval_eval == 0:
            with torch.no_grad():
                after_ocl_preds, after_ocl_gts = run_evaluate_batch(config, model, trainer, ocl_batch)
                after_ocl_score = eval_pt_ds.compute_score_single(after_ocl_gts[0], after_ocl_preds[0])

                before_ocl_score = ocl_iscores[ocl_idx] if ocl_iscores else -1
                ocl_log[ocl_idx] = {'before': before_ocl_score, 'after': after_ocl_score}
                after_ocl_preds_log[ocl_idx] = after_ocl_preds

                after_pt_iscores, after_pt_preds = stat_scores_on_ds(config, model, trainer, eval_pt_ds, return_preds=True)
                after_all_pt_preds_log[ocl_idx] = after_pt_preds

                after_pt_errors = iscores2errors(after_pt_iscores)

                # get replayed vs others scores
                #before_replayed_errors, before_others_errors = stat_replayed_vs_others_scores(replayed_idxs, before_pt_iscores)
                #after_replayed_errors, after_others_errors = stat_replayed_vs_others_scores(replayed_idxs, after_pt_iscores)

                if not config.continual_update:
                    model.load_state_dict(model_state)
                    optimizer.load_state_dict(optim_state)

                aff_log[ocl_idx] = after_pt_errors
                aff_score_log[ocl_idx] = after_pt_iscores
                #aff_loss_log[ocl_idx] = after_pt_losses
                #aff_cl_log[ocl_idx] = {'replayed_idxs': replayed_idxs, 'before_replayed_errors': before_replayed_errors, 'after_replayed_errors': after_replayed_errors,
                #                       'before_others_errors': before_others_errors, 'after_others_errors': after_others_errors}

        with open(os.path.join(config.output_dir, 'aff_log.pkl'),'wb') as wf:
            pickle.dump(aff_log, wf)
        #with open(os.interpath.join(config.output_dir, 'aff_loss_log.pkl'),'wb') as wf:
        #    pickle.dump(aff_loss_log, wf)
        with open(os.path.join(config.output_dir, 'ocl_log.pkl'),'wb') as wf:
            pickle.dump(ocl_log, wf)
        with open(os.path.join(config.output_dir, 'aff_score_log.pkl'),'wb') as wf:
            pickle.dump(aff_score_log, wf)
        with open(os.path.join(config.output_dir, 'after_ocl_preds_log.pkl'),'wb') as wf:
            pickle.dump(after_ocl_preds_log, wf)
        with open(os.path.join(config.output_dir, 'aff_cl_log.pkl'),'wb') as wf:
            pickle.dump(aff_cl_log, wf)
        with open(os.path.join(config.output_dir, 'after_all_pt_preds_log.pkl'),'wb') as wf:
            pickle.dump(after_all_pt_preds_log, wf)

        if step == args.max_step:
            break

    # evaluate on test set
    if args.do_test_eval:
        evaluate_on_dev_set(args, config, model, trainer, tokenizer, 'final')

    if config.continual_update:
        torch.save(model.state_dict(),
                   os.path.join(config.output_dir, 'continual_model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument("--ocl_task")
    parser.add_argument("--max_step", type=int, default=100)
    parser.add_argument("--gc", action='store_true')
    parser.add_argument("--skip_before_eval", action='store_true')
    parser.add_argument("--only_before_eval", action='store_true')
    parser.add_argument("--do_test_eval", action='store_true')
    parser.add_argument("--no_intermed_eval", action='store_true')
    parser.add_argument("--interval_eval", default=1, type=int)
    parser.add_argument("--load_base_ckpt")
    parser.add_argument("--eval_ocl_only", action='store_true')

    args = parser.parse_args()
    run_pipeline_stat_errors(args)
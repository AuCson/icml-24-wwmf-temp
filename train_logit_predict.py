from utils.analysis_tools import initialize, get_base_optimizer, save_model_state, get_logit_batch
from utils.config import merge_config_into_args
import argparse
from transformers import TrainingArguments
from trainer.fgt_prediction_trainer import ForgettingPredictionModel, ForgetPredictionModelForCausualLM
from trainer.fpd_pred_logit_reduce import get_all_alt_scores, reduce_scores, get_pred_grid_reduce, get_pred_grid_em
from data_utils.fpd_helper import FpdP3Helper, DataCollatorWithPaddingStrForFpd, handle_mtl
import logging
import os
import torch
from tqdm import tqdm
import numpy as np

logger = logging.getLogger('fpd_main')

def train_fpd_model(config, fpd_model, fpd_optimizer, fpd_helper):
    fpd_model.train()
    fpd_train_step = config.fpd.train_step
    bs = config.fpd.train_batch_size

    best_score = -1
    key_met = 'f1_mean'
    best_state, best_step = {}, -1
    save_step = 1000
    ckpt_step = config.fpd.ckpt_step
    fpd_optimizer.zero_grad()

    model_state, optim_state = save_model_state(base_model, base_optimizer)

    for step in range(fpd_train_step):
        if (step + 1) % args.eval_step == 0 or (step == 0 and not args.skip_first_eval):
            if config.fpd.method == 'rep_pairwise':
                train_met, _ = infer_fpd_model(config, fpd_model, fpd_helper, 'train')
                val_met, _ = infer_fpd_model(config, fpd_model, fpd_helper, 'dev')
            elif config.fpd.method == 'rep_task_level':
                train_met = None
                #train_met, _ = infer_fpd_model_rep_task(config, fpd_model, fpd_helper, 'train')
                val_met, _ = infer_fpd_model_rep_task(config, fpd_model, fpd_helper, 'dev')
            else:
                train_met, _ = infer_fpd_model_logit_based(config, fpd_model, fpd_helper, 'train')
                val_met, _ = infer_fpd_model_logit_based(config, fpd_model, fpd_helper, 'dev')

            if val_met[key_met] > best_score:
                best_state = {k:v.cpu().clone() for k,v in fpd_model.state_dict().items()}
                best_score = val_met[key_met]
                best_step = step + 1

        if config.fpd.method == 'rep_pairwise':
            fpd_batch = fpd_helper.sample_episode_batch_paired('train', bs=bs)
            fpd_batch = fpd_model.clean_batch(fpd_batch)
            pred_logits, loss = fpd_model.pred_forget_pairwise(**fpd_batch)
        elif config.fpd.method == 'rep_task_level':
            fpd_batch = fpd_helper.sample_episode_task_level_balanced('train',bs=bs)
            fpd_batch = fpd_model.batch_to_cuda(fpd_batch)
            if config.fpd.binarilize_labels:
                pred_logits, loss = fpd_model.pred_forget_pairwise_ce(**fpd_batch)
            else:
                pred_logits, loss = fpd_model.pred_forget_pairwise_mse(**fpd_batch)
        elif config.fpd.method == 'logit_pairwise':
            fpd_model.train(False)
            extra_infos = ['logits_change']
            if config.fpd.logit_loss_type == 'kl':
                extra_infos.append('pt_logits_change')
            fpd_batch = fpd_helper.sample_episode_batch_paired('train', bs=bs, extra_infos=extra_infos)
            fpd_batch = fpd_model.clean_batch(fpd_batch)
            if config.fpd.multi_token:
                pred_logits, loss, loss_vec = fpd_model.pred_forget_logit_based_multi(**fpd_batch)
            else:
                pred_logits, loss, loss_vec = fpd_model.pred_forget_logit_based(**fpd_batch)
        elif config.fpd.method == 'logit_direct':
            fpd_model.train(False)
            ocl_batch, pt_batch = fpd_helper.sample_episode_random_ocl('train', bs=bs)
            ocl_batch, pt_batch = base_trainer.batch_to_gpu(ocl_batch), base_trainer.batch_to_gpu(pt_batch)
            with torch.no_grad():
                before_ocl_logits = get_logit_batch(config, base_model, base_trainer, ocl_batch)
                before_pt_logits = get_logit_batch(config, base_model, base_trainer, pt_batch)
            base_trainer.nstep_model_update(base_model, ocl_batch, base_optimizer, n_step=config.ocl_steps, eval_mode=True)
            with torch.no_grad():
                after_ocl_logits = get_logit_batch(config, base_model, base_trainer, ocl_batch)
                after_pt_logits = get_logit_batch(config, base_model, base_trainer, pt_batch)
            ocl_logit_change = after_ocl_logits - before_ocl_logits

            pred_logits, loss = fpd_model.pred_directly_logit_change(ocl_batch['input_ids'], ocl_batch['attention_mask'],
                                                                     ocl_batch['labels'], ocl_batch['_decoder_attention_mask'],
                                                                     pt_batch['input_ids'],
                                                                     pt_batch['attention_mask'],
                                                                     pt_batch['labels'],
                                                                     pt_batch['_decoder_attention_mask'],
                                                    ocl_logit_change, before_pt_logits, after_pt_logits)

            base_model.load_state_dict(model_state)
            base_optimizer.load_state_dict(optim_state)

        else:
            raise NotImplementedError

        logger.info(f'Training loss, step {step}, loss {loss.item()}')

        loss = loss / config.fpd.grad_accum
        loss.backward()
        if (step + 1) % config.fpd.grad_accum == 0:
            fpd_optimizer.step()
            fpd_optimizer.zero_grad()

        if (step + 1) % save_step == 0:
            torch.save({'step': best_step, 'config': config.to_dict(), 'state': best_state, 'score': best_score},
                       os.path.join(config.output_dir, 'best_model.pt'))

        if ckpt_step > 0 and (step + 1) % ckpt_step == 0:
            torch.save({'step': step, 'config': config.to_dict(), 'state': {k:v.cpu().clone() for k,v in fpd_model.state_dict().items()}},
                       os.path.join(config.output_dir, 'model.{}.pt'.format(step + 1)))

def infer_fpd_model(config, fpd_model, fpd_helper: FpdP3Helper, split, save_path=None, try_thres=False):
    print('Starting inference')
    is_training = fpd_model.training
    fpd_model.eval()
    # get reps of all pt and ocl_examples
    pt_loader, ocl_loader = fpd_helper.get_pt_dataloader(split, config.fpd.eval_batch_size), \
                            fpd_helper.get_ocl_dataloader(split, config.fpd.eval_batch_size)
    fgt_label_grid, base_error_mask = fpd_helper.get_label_grid_and_base_error(split)

    all_priors = fpd_helper.get_all_priors('train')
    if all_priors is not None: all_priors = all_priors.cuda()

    # all reps
    all_pt_reps = []
    all_ocl_reps = []

    with torch.no_grad():
        print('Getting PT example reps')
        for _, pt_batch in tqdm(enumerate(pt_loader), total=len(pt_loader)):
            pt_batch = fpd_model.clean_batch_for_rep(pt_batch)
            reps = fpd_model.get_reps(pt_batch['input_ids'], pt_batch['attention_mask'], pt_batch['labels'],
                                      pt_batch['decoder_attention_mask'])
            reps = reps.detach()
            all_pt_reps.append(reps)
        all_pt_reps = torch.cat(all_pt_reps, 0) # [N1,H]
        print('Getting OCL example reps')
        for _, ocl_batch in tqdm(enumerate(ocl_loader), total=len(ocl_loader)):
            ocl_batch = fpd_model.clean_batch_for_rep(ocl_batch)
            reps = fpd_model.get_reps(ocl_batch['input_ids'], ocl_batch['attention_mask'], ocl_batch['labels'],
                                      ocl_batch['decoder_attention_mask'])
            reps = reps.detach()
            all_ocl_reps.append(reps)
        all_ocl_reps = torch.cat(all_ocl_reps, 0) # [N2, H]

        if try_thres:
            for thres in np.arange(0.4,1,0.005):
                prob_grid, preds_grid = fpd_model.pred_forget_with_reps(all_ocl_reps, all_pt_reps,
                                                                        all_priors=all_priors, thres=thres)
                met_dict = fpd_helper.evaluate_metrics(fgt_label_grid, preds_grid, base_error_mask)
                logger.info('Thres {} Metrics over {}: {}'.format(thres, split, met_dict))

        prob_grid, preds_grid = fpd_model.pred_forget_with_reps(all_ocl_reps, all_pt_reps,
                                                                all_priors=all_priors)
        met_dict = fpd_helper.evaluate_metrics(fgt_label_grid, preds_grid, base_error_mask)
    logger.info('Metrics over {}: {}'.format(split, met_dict))
    fpd_model.train(is_training)
    if save_path:
        print('Save path is', save_path)
        fpd_helper.save_preds(preds_grid, prob_grid, base_error_mask, save_path, split)
    return met_dict, preds_grid


def infer_fpd_model_rep_task(config, fpd_model, fpd_helper: FpdP3Helper, split, save_path=None, try_thres=False):
    print('Starting inference')
    is_training = fpd_model.training
    fpd_model.eval()
    # get reps of all pt and ocl_examples
    pt_loader, ocl_loader = fpd_helper.get_pt_dataloader(split, config.fpd.eval_batch_size), \
                            fpd_helper.get_ocl_dataloader_concat(split, config.fpd.eval_batch_size)
    fgt_label_grid, base_error_mask = fpd_helper.get_label_grid_and_base_error_binarilize(split)

    # all reps
    all_pt_reps = []
    all_ocl_reps = []

    with torch.no_grad():
        print('Getting PT example reps')
        for idx, pt_batch in tqdm(enumerate(pt_loader), total=len(pt_loader)):
            pt_batch = fpd_model.batch_to_cuda(pt_batch)
            reps = fpd_model.get_reps(pt_batch['input_ids'])
            reps = reps.detach()
            all_pt_reps.append(reps)
            #if idx == 10:
            #    break
        all_pt_reps = torch.cat(all_pt_reps, 0) # [N1,H]
        print('Getting OCL example reps')
        for idx, ocl_batch in tqdm(enumerate(ocl_loader), total=len(ocl_loader)):
            ocl_batch = fpd_model.batch_to_cuda(ocl_batch)
            reps = fpd_model.get_reps(ocl_batch['input_ids'])
            reps = reps.detach()
            all_ocl_reps.append(reps)
            #if idx == 10:
            #    break
        all_ocl_reps = torch.cat(all_ocl_reps, 0) # [N2, H]
        prod_mat = fpd_model.get_rep_prod_mat(all_ocl_reps, all_pt_reps)
        prob_grid, preds_grid = fpd_model.pred_forget_with_reps_score(all_ocl_reps, all_pt_reps, thres=0.)
        met_dict = fpd_helper.evaluate_metrics(fgt_label_grid, preds_grid, base_error_mask)
    logger.info('Metrics over {}: {}'.format(split, met_dict))
    fpd_model.train(is_training)
    if save_path:
        print('Save path is', save_path)
        fpd_helper.save_raw_scores(prod_mat, save_path, split)
        fpd_helper.save_preds(preds_grid, prob_grid, base_error_mask, save_path, split)
    return met_dict, preds_grid


def infer_fpd_model_logit_based(config, fpd_model, fpd_helper: FpdP3Helper, split, save_path=None):
    print('Starting inference')
    is_training = fpd_model.training
    fpd_model.eval()
    # get reps of all pt and ocl_examples
    pt_loader, ocl_loader = fpd_helper.get_pt_dataloader(split, config.fpd.eval_batch_size), \
                            fpd_helper.get_ocl_dataloader(split, 1)
    pt_ds = fpd_helper.get_pt_ds(split)
    fgt_label_grid, base_error_mask = fpd_helper.get_label_grid_and_base_error(split)

    all_pt_logits = fpd_helper.get_all_pt_logits(split)
    all_ocl_logits_change = fpd_helper.get_all_ocl_logits_change(split)
    #all_gt_forgets = fpd_helper.get_all_gt_forgets(split)

    # all reps
    all_pt_reps = []
    all_ocl_reps = []

    all_pt_dec_attn_masks = []
    all_ocl_dec_attn_masks = []

    preds_grid = []
    is_all_ts = config.fpd.multi_token

    all_pred_pt_logits, all_pred_pt_logits_idxs = [], []

    with torch.no_grad():
        print('Getting PT example reps')
        for _, pt_batch in tqdm(enumerate(pt_loader), total=len(pt_loader)):
            pt_batch = fpd_model.clean_batch_for_rep(pt_batch)
            reps = fpd_model.get_reps(pt_batch['input_ids'], pt_batch['attention_mask'], pt_batch['labels'],
                                      pt_batch['decoder_attention_mask'], all_ts=is_all_ts)
            reps = reps.detach()
            all_pt_reps.append(reps)
            all_pt_dec_attn_masks.append(pt_batch['decoder_attention_mask'])

        if is_all_ts:
            all_pt_reps = fpd_model.pad_and_cat(all_pt_reps) # [N1,H] [N1,T,H]
            all_pt_dec_attn_masks = fpd_model.pad_and_cat_attn(all_pt_dec_attn_masks)
        else:
            all_pt_reps = torch.cat(all_pt_reps, 0)  # [N1,H] [N1,T,H]
        print('Getting OCL example reps')

        for _, ocl_batch in tqdm(enumerate(ocl_loader), total=len(ocl_loader)):
            ocl_batch = fpd_model.clean_batch_for_rep(ocl_batch)
            reps = fpd_model.get_reps(ocl_batch['input_ids'], ocl_batch['attention_mask'], ocl_batch['labels'],
                                      ocl_batch['decoder_attention_mask'], all_ts=is_all_ts)
            reps = reps.detach()
            all_ocl_reps.append(reps)
            all_ocl_dec_attn_masks.append(ocl_batch['decoder_attention_mask'])

        if not is_all_ts:
            all_ocl_reps = torch.cat(all_ocl_reps, 0) # [N2, H] [N2,T,H]

        for ocl_error_idx in tqdm(range(len(all_ocl_reps)), total=len(all_ocl_reps)):

            ocl_logits_change = all_ocl_logits_change[ocl_error_idx]
            ocl_reps = all_ocl_reps[ocl_error_idx]
            ocl_dec_attn_mask = all_ocl_dec_attn_masks[ocl_error_idx]

            if config.fpd.multi_token:
                if args.return_pred_logits or config.fpd.late_pred_forget:
                    preds_forget, pred_pt_logits, pred_pt_logits_idxs = fpd_model.infer_pred_forget_with_reps_logit_multi(
                                                                                     ocl_reps, all_pt_reps, all_pt_logits,
                                                                                     ocl_logits_change,ocl_dec_attn_mask,
                                                                                     all_pt_dec_attn_masks,
                                                                                     return_pred_logits=True,
                                                                                     skip_pred=config.fpd.late_pred_forget)
                    all_pred_pt_logits.append(pred_pt_logits)
                    all_pred_pt_logits_idxs.append(pred_pt_logits_idxs)
                else:
                    preds_forget = fpd_model.infer_pred_forget_with_reps_logit_multi(ocl_reps, all_pt_reps,
                                                                                     all_pt_logits, ocl_logits_change,
                                                                                     ocl_dec_attn_mask,
                                                                                     all_pt_dec_attn_masks)



            else:
                preds_forget = fpd_model.infer_pred_forget_with_reps_logit_single(ocl_reps, all_pt_reps, all_pt_logits,
                                                                                  ocl_logits_change)

            if not config.fpd.late_pred_forget:
                preds_arr = torch.zeros(fgt_label_grid.size(1)).long()
                preds_arr[preds_forget] = 1
                preds_grid.append(preds_arr)

    if config.fpd.late_pred_forget:
        pred_logits_obj = {
            'pred_logits': all_pred_pt_logits,
            'pred_logits_idxs': all_pred_pt_logits_idxs
            }
        if config.fpd.reduce_method != 'exact':
            thres = 0
            alt_scores_info = get_all_alt_scores(config, pred_logits_obj, pt_ds, tokenizer)
            label_score_grid, alt_score_grid = get_pred_grid_reduce(alt_scores_info, pt_ds, config, method=config.fpd.reduce_method)
            preds_grid = (alt_score_grid - label_score_grid) > thres
            preds_grid = torch.from_numpy(preds_grid)
        else:
            preds_grid = get_pred_grid_em(config, all_pred_pt_logits, all_pred_pt_logits_idxs, pt_ds, tokenizer)
            preds_grid = torch.from_numpy(preds_grid)

    #print(fgt_label_grid, preds_grid, base_error_mask)
    if type(preds_grid) is list:
        preds_grid = torch.stack(preds_grid)
    met_dict = fpd_helper.evaluate_metrics(fgt_label_grid, preds_grid, base_error_mask)

    logger.info('Metrics over {}: {}'.format(split, met_dict))
    fpd_model.train(is_training)
    if save_path:
        fpd_helper.save_preds(preds_grid, np.zeros_like(preds_grid), base_error_mask, save_path, split)
    if args.return_pred_logits:
        if save_path is None:
            save_path = os.path.join(config.output_dir,f'fpd_dev/{args.ocl_task}')
        fpd_helper.save_pred_logits(all_pred_pt_logits, all_pred_pt_logits_idxs, save_path, split)

    return met_dict, preds_grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')
    parser.add_argument("--ocl_task")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--load_model_dir")
    parser.add_argument("--load_model_name", default='best_model.pt')
    parser.add_argument("--skip_first_eval", action='store_true')
    parser.add_argument("--return_pred_logits", action='store_true')
    parser.add_argument("--try_thres", action='store_true')

    args = parser.parse_args()

    config, base_model, tokenizer, base_trainer, collator = initialize(args.config_files, args.templates)
    base_model = base_model.cuda()

    # mtl or not
    if config.fpd.mtl:
        mtl_tasks_str = config.fpd.mtl_tasks
        mtl_tasks = mtl_tasks_str.split('+')
        config = handle_mtl(config, mtl_tasks) # output dir will be updated

    training_args = TrainingArguments(output_dir=config.output_dir)
    merge_config_into_args(config, training_args)

    base_optimizer = get_base_optimizer(config, base_model, training_args=training_args)


    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    config.save(config.output_dir, 'config.json')

    fpd_collator = DataCollatorWithPaddingStrForFpd(tokenizer)


    fpd_helper = FpdP3Helper(config, tokenizer, fpd_collator, args.ocl_task)

    if config.is_seq2seq:
        fpd_model = ForgettingPredictionModel(config, tokenizer, fpd_helper).cuda()
    else:
        fpd_model = ForgetPredictionModelForCausualLM(config, tokenizer, fpd_helper).cuda()

    fpd_optimizer = fpd_model.create_optimizer()

    if args.load_model_dir:
        load_model_dir = args.load_model_dir
        print('Loading from {}'.format(load_model_dir))

        model_dir = os.path.join(load_model_dir, args.load_model_name)
        save_obj = torch.load(model_dir)
        fpd_model.load_state_dict(save_obj['state'])

    if args.do_train:
        train_fpd_model(config, fpd_model, fpd_optimizer, fpd_helper)

    if args.do_eval:
        if not args.load_model_dir:
            load_model_dir = config.output_dir
        else:
            load_model_dir = args.load_model_dir
        print('Loading from {}'.format(load_model_dir))
        model_dir = os.path.join(load_model_dir, args.load_model_name)

        if not os.path.isfile(model_dir):
            logger.info("No trained model found. Evaluating with fresh model.")
        else:
            save_obj = torch.load(model_dir)
            fpd_model.load_state_dict(save_obj['state'])

        if config.fpd.method == 'rep_pairwise':
            met_dict, preds_grid = infer_fpd_model(config, fpd_model, fpd_helper, split='dev',
                                                   save_path=os.path.join(config.output_dir,f'fpd_dev/{args.ocl_task}'),
                                                   try_thres=args.try_thres)
        elif config.fpd.method == 'logit_pairwise':
            met_dict, preds_grid = infer_fpd_model_logit_based(config, fpd_model, fpd_helper, split='dev',
                                                               save_path=os.path.join(config.output_dir,f'fpd_dev/{args.ocl_task}'))
        elif config.fpd.method == 'rep_task_level':
            met_dict, preds_grid = infer_fpd_model_rep_task(config, fpd_model, fpd_helper, 'dev',
                                                  save_path=os.path.join(config.output_dir, f'fpd_dev/{args.ocl_task}'))
        else:
            raise NotImplementedError(config.fpd.method)
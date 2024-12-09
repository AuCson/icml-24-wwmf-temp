from transformers.trainer import Trainer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from .utils import trim_batch
import logging
import os

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    pass

dist_logger = logging.getLogger('dist')


def count_flops(model, inputs, name):
    flops = FlopCountAnalysis(model, inputs)
    total = flops.total()
    dist_logger.info('Flop of {}: {}'.format(name, total))
    return total



class ForgettingPredictionTrainer(Trainer):
    def __init__(self, **kwargs):
        self.base_model = kwargs.pop('base_model')
        self.base_trainer = kwargs.pop('base_trainer')
        super().__init__(**kwargs)


class ContrastiveHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class VocabMap(nn.Module):
    def __init__(self, lm):
        super().__init__()
        data = lm.lm_head.weight.detach().clone() # [V,H]
        self.embed = nn.Parameter(data)

    def get_vocab_map(self):
        mat = torch.matmul(self.embed, self.embed.transpose(0,1)) / self.embed.size(1) # [V,V]
        return mat

    def forward(self, pred_change): # [B,V]
        x = torch.matmul(pred_change, self.embed)# [B,H]
        ret = torch.matmul(x, self.embed.transpose(0,1)) / self.embed.size(1) # [B,V]
        return ret

class LogitFpdForwardModule(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, rep_dist_i, ocl_logits_change):
        pred_logit_update = torch.matmul(rep_dist_i / rep_dist_i.size(1), ocl_logits_change)  # [T1,V]
        return pred_logit_update

class ForgettingPredictionModel(nn.Module):
    def __init__(self, config, tokenizer, helper, init_model=True):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.use_head = config.fpd.use_head
        self.use_vocab_map = config.fpd.vocab_map
        self.normalize = config.fpd.normalize
        self.temp = config.fpd.temp
        self.head = None
        self.vocab_map = None
        self.helper = helper

        if init_model:
            self.lm = AutoModelForSeq2SeqLM.from_pretrained(config.fpd.model_name)
            if self.use_head:
                self.head = ContrastiveHead(getattr(self.lm.config, 'd_model', self.lm.config.hidden_size), config.fpd.output_dim)
            if self.use_vocab_map:
                self.vocab_map = VocabMap(self.lm)
        dist_logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'dist_log.txt')))

    def create_optimizer(self):
        param_groups = []
        if not self.config.fpd.freeze_lm:
            param_groups.append({'params': self.lm.parameters()})
        if self.head is not None:
            param_groups.append({'params': self.head.parameters(), 'lr': self.config.fpd.lr * self.config.fpd.lr_scale})
        if self.use_vocab_map:
            param_groups.append({'params': self.vocab_map.parameters(), 'lr': self.config.fpd.lr})

        optimizer = torch.optim.Adam(
            param_groups,
            lr=self.config.fpd.lr
        )
        return optimizer

    def forward(self, mode, *inputs):
        # for flop analysis only
        if mode == 'get_reps':
            return self.get_reps(*inputs)
        elif mode == 'infer_pred_forget_with_reps_logit_single':
            return self.infer_pred_forget_with_reps_logit_single(*inputs)
        elif mode == 'pred_forget_with_reps':
            return self.pred_forget_with_reps(*inputs)
        elif mode == 'infer_pred_forget_with_reps_logit_single':
            return self.infer_pred_forget_with_reps_logit_single_profile(*inputs)
        else:
            raise ValueError(mode)

    def get_reps(self, input_ids, attention_mask, labels, decoder_attention_mask, all_ts=False):

        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                          output_hidden_states=True)
        decoder_input_len = decoder_attention_mask.sum(-1) # [B]
        if all_ts:
            raw_rep = outputs.decoder_hidden_states[-1] # [B,T,H]
        else:
            last_layer_hidden = outputs.decoder_hidden_states[-1] # [B,T,H]
            raw_rep = last_layer_hidden[torch.arange(last_layer_hidden.size(0)),decoder_input_len - 1,:]

        if self.config.fpd.freeze_lm:
            raw_rep = raw_rep.detach()

        if self.use_head:
            rep = self.head(raw_rep)
        else:
            rep = raw_rep

        if self.normalize:
            rep = F.normalize(rep, dim=-1)

        return rep

    def get_rep_prod(self, rep_a, rep_b):
        #print(rep_a, rep_b)
        # [B,H], [B,H]
        if self.config.fpd.use_cos_dist:
            rep_prod = F.cosine_similarity(rep_a, rep_b, dim=-1) # [B,T1*T2]
        else:
            if self.config.fpd.sum_or_mean == 'sum':
                rep_prod = (rep_a * rep_b).sum(-1)
            else:
                rep_prod = (rep_a * rep_b).mean(-1)
        rep_prod = self.config.rep_prod_sgn * rep_prod / self.temp
        return rep_prod

    def get_rep_prod_mat(self, all_ocl_reps, all_pt_reps):
        # [N2, H], [N1,H]
        if self.config.fpd.use_cos_dist:
            assert all_ocl_reps.size(0) == 1
            rep_prod_grid = F.cosine_similarity(all_ocl_reps, all_pt_reps) # [NT1]
            rep_prod_grid = rep_prod_grid.unsqueeze(0)
        else:
            if self.config.fpd.sum_or_mean == 'sum':
                rep_prod_grid = torch.matmul(all_ocl_reps, all_pt_reps.transpose(0,1))
            else:
                rep_prod_grid = torch.matmul(all_ocl_reps, all_pt_reps.transpose(0,1)) / float(all_ocl_reps.size(1))
        rep_prod_grid = self.config.rep_prod_sgn * rep_prod_grid / self.temp 
        return rep_prod_grid


    def pred_forget_pairwise(self, input_ids_pt, input_ids_ocl, attention_mask_pt, attention_mask_ocl, labels_pt, labels_ocl,
                decoder_attention_mask_pt, decoder_attention_mask_ocl, priors=None, forget_label=None, **kwargs):
        rep_a = self.get_reps(input_ids_pt, attention_mask_pt, labels_pt, decoder_attention_mask_pt)
        rep_b = self.get_reps(input_ids_ocl, attention_mask_ocl, labels_ocl, decoder_attention_mask_ocl)
        logit = self.get_rep_prod(rep_a, rep_b)

        logit = self.add_bias_to_logit_if_needed(logit, priors)

        prob = F.sigmoid(logit)
        loss = None
        #print(forget_label, prob)

        weights = torch.where(forget_label == 1, self.config.fpd.ce_loss_pos_weight, 1.)

        if forget_label is not None:
            loss = F.binary_cross_entropy(prob, forget_label.float(), weight=weights)
        return prob, loss

    def pred_forget_with_reps(self, all_ocl_reps, all_pt_reps, all_priors, thres=0.5):
        logits = self.get_rep_prod_mat(all_ocl_reps, all_pt_reps)
        logits = self.add_bias_to_logit_if_needed(logits, all_priors)

        prob_grid = F.sigmoid(logits) # [N2,N1]

        preds = (prob_grid > thres).long()
        #print(prob_grid, preds)
        return prob_grid, preds

    def pred_directly_logit_change(self, input_ids_ocl, attention_mask_ocl, labels_ocl, decoder_attention_mask_ocl,
                                   input_ids_pt, attention_mask_pt, labels_pt, decoder_attention_mask_pt,
                                   ocl_logit_change, pt_logits_before, pt_logits_after):
        # ocl is a single example batch here

        rep_a = self.get_reps(input_ids_pt, attention_mask_pt, labels_pt, decoder_attention_mask_pt, all_ts=True) # [B,T1,H] 123 -> 111122223333
        rep_b_single = self.get_reps(input_ids_ocl, attention_mask_ocl, labels_ocl, decoder_attention_mask_ocl, all_ts=True) # [1,T2,H]  _> 123412341234

        rep_b = rep_b_single.repeat(rep_a.size(0), 1, 1)

        # cosine similarity or simply dot product?
        rep_a_xxyy = torch.repeat_interleave(rep_a, rep_b.size(1), dim=1) # [B,T1*T2, H]
        rep_b_xyxy = rep_b.repeat(1, rep_a.size(1), 1)
        #rep_dists = F.cosine_similarity(rep_a_xxyy, rep_b_xyxy, dim=-1) # [B,T1*T2]
        rep_dists = self.get_rep_prod(rep_a_xxyy, rep_b_xyxy)
        rep_dists = rep_dists.view(rep_a.size(0), rep_a.size(1), rep_b.size(1)) # [B,T1,T2]

        rep_dists_mask = self.get_rep_dists_mask(decoder_attention_mask_pt, decoder_attention_mask_ocl.repeat(rep_a.size(0), 1))
        rep_dists = rep_dists * rep_dists_mask.float()

        pred_logits_update = torch.matmul(rep_dists / rep_dists.size(-1), ocl_logit_change)  # [B,T1,V] # normalized by T2 # bugfix 0118
        pred_logits = pred_logits_update + pt_logits_before  # [B,T1,V]

        if self.config.fpd.direct_loss_type == 'kl':
            loss = self.simple_kl_loss(pt_logits_after, pred_logits)
        elif self.config.fpd.direct_loss_type == 'ce':
            loss = self.hard_ce_loss(pt_logits_after, pred_logits, labels_pt)
        return pred_logits, loss

    def pred_forget_logit_based(self, input_ids_pt, input_ids_ocl, attention_mask_pt, attention_mask_ocl, labels_pt, labels_ocl,
                decoder_attention_mask_pt, decoder_attention_mask_ocl, ocl_update_logits, pt_logits_ss, pt_logits_idxs, priors=None,
                forget_label=None, **kwargs):
        rep_a = self.get_reps(input_ids_pt, attention_mask_pt, labels_pt, decoder_attention_mask_pt)
        rep_b = self.get_reps(input_ids_ocl, attention_mask_ocl, labels_ocl, decoder_attention_mask_ocl)

        rep_dist = F.cosine_similarity(rep_a, rep_b) # [B]
        #rep_dist = self.get_rep_prod(rep_a, rep_b) #

        # pt_logits_ss, pt_logits_idxs: [B, T, Vs]
        vocab_size = ocl_update_logits.size(-1)

        pt_logits_full = torch.full((pt_logits_ss.size(0), pt_logits_ss.size(1), vocab_size), -100.0).to(pt_logits_ss.device)
        pt_logits_mask = torch.zeros(pt_logits_ss.size(0), pt_logits_ss.size(1), vocab_size, dtype=torch.bool).to(pt_logits_ss.device)

        pt_logits_full.scatter_(2, pt_logits_idxs, pt_logits_ss) # [B,T,V]
        pt_logits_mask.scatter_(2, pt_logits_idxs, 1)

        pred_logits_update = rep_dist.view(rep_dist.size(0), 1) * ocl_update_logits[:,self.config.fpd.ts,:] # [B,V]
        pred_logits = pred_logits_update.unsqueeze(1) + pt_logits_full # [B,T,V]

        pred_logits_masked = pred_logits.masked_fill_(~pt_logits_mask, -np.inf)

        loss, loss_vec = None, None
        if forget_label is not None:
            #if self.config.fpd.loss_type == 'default':
            ce = F.cross_entropy(pred_logits_masked.view(-1, pred_logits_masked.size(2)), labels_pt.view(-1), reduction='none')
            ce = (ce.view(pred_logits_masked.size(0),
                          pred_logits_masked.size(1)) * decoder_attention_mask_pt.float()).mean(-1) # [B]
            sgn = torch.where(forget_label == 1, -1, 1)
            loss_vec = ce * sgn
            loss = loss_vec.mean()

        return pred_logits_masked, loss, loss_vec

    def masked_kl_loss(self, pred_logits, ref_logits, logit_mask, ref_mask, labels_pt, pt_logits_before):
        # [B,T,V], [B,T,V]
        ts_mask = (labels_pt != -100).unsqueeze(-1).expand(-1,-1, pred_logits.size(2)) # [B,T,V]

        if self.config.fpd.logit_ocl_ts_only:
            ts_sel_mask = torch.zeros_like(ts_mask)
            ts_sel_mask[:,self.config.fpd.ts] = 1
            ts_mask = ts_mask & ts_sel_mask

        if self.config.fpd.ignore_zero_token_kl:
            ts_mask[:,:,0] = 0
            logit_mask[:,:,0] = 0
            ref_mask[:,:,0] = 0

        joint_mask = logit_mask & ref_mask

        pred_logits_masked = pred_logits.masked_fill(~joint_mask, -np.inf)
        ref_logits_masked = ref_logits.masked_fill(~joint_mask, -np.inf)

        if self.config.fpd.use_mse_variant:
            l2_diff = (pred_logits_masked - ref_logits_masked) ** 2
            l2_diff = l2_diff.masked_fill(~joint_mask, 0) # [B,T,V]
            l2_diff_ts_masked = l2_diff[ts_mask]
            l2_diff = l2_diff_ts_masked.sum() / (ts_mask.float().sum() / 100)
            return l2_diff
        else:
            ref_prob = F.softmax(ref_logits_masked, -1) # [B,T,V]
            ref_log_prob = torch.log(ref_prob)
            pred_log_prob = F.log_softmax(pred_logits_masked, -1)
            log_prob_diff = ref_log_prob - pred_log_prob
            log_prob_diff = log_prob_diff.masked_fill(~joint_mask, 0) # [B,T,V]

            kl_div = ref_prob * log_prob_diff # [B,T,V]
            kl_div_ts_masked = kl_div[ts_mask]
            kl_div = kl_div_ts_masked.sum() / (ts_mask.float().sum() / ts_mask.size(2))
            return kl_div

    def simple_kl_loss(self, ref_logits, pred_logits):
        ref_prob = F.softmax(ref_logits, -1)  # [B,T,V]
        ref_log_prob = F.log_softmax(ref_logits, -1)
        pred_log_prob = F.log_softmax(pred_logits, -1)
        log_prob_diff = ref_log_prob - pred_log_prob
        kl_div = ref_prob * log_prob_diff  # [B,T,V]
        kl_div = kl_div.sum() / (kl_div.size(0) * kl_div.size(1))
        return kl_div

    def weighted_kl_loss(self, ref_logits, pred_logits, before_label):
        ref_label = ref_logits.max(-1)[1]
        mask = (ref_label != before_label) & (before_label != -100)
        weight = torch.where(mask, 10.0, 0.1) # [B,T]

        ref_prob = F.softmax(ref_logits, -1)  # [B,T,V]
        ref_log_prob = F.log_softmax(ref_logits, -1)
        pred_log_prob = F.log_softmax(pred_logits, -1)
        log_prob_diff = ref_log_prob - pred_log_prob
        kl_div = ref_prob * log_prob_diff * weight.unsqueeze(-1)  # [B,T,V]
        kl_div = kl_div.sum() / (kl_div.size(0) * kl_div.size(1))
        return kl_div

    def hard_ce_loss(self, ref_logits, pred_logits, before_label):
        ref_label_ = ref_logits.max(-1)[1]
        ref_label = torch.where(before_label != -100, ref_label_, -100)
        loss = F.cross_entropy(pred_logits.view(-1, pred_logits.size(-1)), ref_label.view(-1))
        return loss

    def get_top_cand_single_ts(self, pred_logits, labels):
        # [B,V], [B]
        topv, topi = pred_logits.topk(10, -1) # [B]
        top_cands = []

        for b in range(labels.size(0)):
            diff_token, diff_sem_token = -1, -1
            label_token = self.tokenizer.convert_ids_to_tokens(labels[b].item())
            for idx in range(len(topi)):
                if diff_token == -1 and labels[b].item() != topi[b,idx].item():
                    diff_token = topi[b,idx].item()
                if label_token.lower() != self.tokenizer.convert_ids_to_tokens(topi[b,idx].item()).lower():
                    diff_sem_token = topi[b,idx].item()
                    break
            if diff_sem_token == -1:
                diff_sem_token = diff_token
            top_cands.append(diff_sem_token)

        top_cand_idxs = torch.LongTensor(top_cands).to(pred_logits.device) # [B]
        return top_cand_idxs

    def get_top_cand_multi_ts(self, pred_logits, labels):
        # [B,T,V], [B,T]
        topv, topi = pred_logits.topk(10, -1) # [B,T,N]
        top_cands = torch.zeros_like(labels) # [B,T]
        top_cand_scores = torch.zeros_like(labels).float()
        for b in range(labels.size(0)):
            for t in range(labels.size(1)):
                label_token_idx = labels[b,t].item()
                diff_token, diff_sem_token = -1, -1
                if label_token_idx != 0 and label_token_idx != -100:
                    label_token = self.tokenizer.convert_ids_to_tokens(label_token_idx)
                    for idx in range(len(topi)):
                        cand_token_idx = topi[b,t,idx].item()
                        cand_token = self.tokenizer.convert_ids_to_tokens(cand_token_idx)
                        if diff_token == -1 and label_token_idx != cand_token_idx:
                            diff_token = cand_token_idx
                        if label_token.lower() != cand_token.lower():
                            diff_sem_token = cand_token_idx
                            break
                    if diff_sem_token == -1:
                        diff_sem_token = diff_token
                    top_cands[b,t] = diff_sem_token
                    top_cand_scores[b,t] = pred_logits[b,t,diff_sem_token]
        return top_cands, top_cand_scores

    def margin_loss(self, pred_logits, logit_mask, labels_pt, fgt_label, ref_mask=None):
        ts_mask = (labels_pt != -100).unsqueeze(-1).expand(-1,-1, pred_logits.size(2)) # [B,T,V]

        assert self.config.fpd.logit_ocl_ts_only

        #if self.config.fpd.logit_ocl_ts_only:
        ts_sel_mask = torch.zeros_like(ts_mask)
        ts_sel_mask[:,self.config.fpd.ts] = 1
        ts_mask = ts_mask & ts_sel_mask

        assert self.config.fpd.ignore_zero_token_kl
        #if self.config.fpd.ignore_zero_token_kl:
        ts_mask[:,:,0] = 0
        logit_mask[:,:,0] = 0
        if ref_mask is not None:
            ref_mask[:,:,0] = 0

        joint_mask = logit_mask & ref_mask if ref_mask is not None else logit_mask

        pred_logits_masked_ts = pred_logits.masked_fill(~joint_mask, -np.inf)[:, self.config.fpd.ts] # [B,V]

        # get top candidate pred
        top_cand_idxs = self.get_top_cand_single_ts(pred_logits_masked_ts, labels_pt[:, self.config.fpd.ts]) # [B]
        top_cand_scores = pred_logits_masked_ts[torch.arange(top_cand_idxs.size(0)), top_cand_idxs]
        label_scores = pred_logits_masked_ts[torch.arange(top_cand_idxs.size(0)), labels_pt[:, self.config.fpd.ts]]

        if torch.any(torch.isinf(label_scores)):
            print('inf in label scores')
            label_scores = label_scores.masked_scatter(torch.isinf(label_scores), top_cand_scores)

        sgn = torch.where(fgt_label == 1, 1, -1)
        diff = top_cand_scores - label_scores

        margin = self.config.fpd.margin_value
        raw_score = sgn * diff - margin
        raw_score = raw_score.masked_fill(raw_score > 0, 0)

        loss = -raw_score

        if self.config.fpd.margin_sq:
            loss = loss ** 2

        if self.config.fpd.margin_loss_reweight:
            coeff = torch.where(fgt_label == 1, 0.1, 1.0)
            loss = coeff * loss

        return loss.mean()

    def margin_loss_multi(self, pred_logits, logit_mask, labels_pt, fgt_label, ref_mask=None):
        ts_mask = (labels_pt != -100).unsqueeze(-1).expand(-1, -1, pred_logits.size(2))  # [B,T,V]

        assert self.config.fpd.ignore_zero_token_kl
        ts_mask[:,:,0] = 0
        logit_mask[:,:,0] = 0
        if ref_mask is not None:
            ref_mask[:,:,0] = 0

        joint_mask = logit_mask & ref_mask if ref_mask is not None else logit_mask
        pred_logits_masked = pred_logits.masked_fill(~joint_mask, -np.inf) # [B,T,V]

        # [B,T,V] [B,T] # s[b][t][v[b,t]]

        top_cand_idxs, top_cand_scores = self.get_top_cand_multi_ts(pred_logits_masked, labels_pt) # [B,T], [B,T]

        label_pt_mask = (labels_pt != -100) & (labels_pt != 0)
        label_pt_clean = labels_pt.masked_fill(~label_pt_mask, 0)
        label_scores = torch.gather(pred_logits_masked, 2, label_pt_clean.unsqueeze(-1)).squeeze(-1) # [B,T]
        label_scores = label_scores.masked_scatter(torch.isinf(label_scores), top_cand_scores)

        sgn = torch.where(fgt_label == 1, 1, -1).unsqueeze(-1).expand(-1, labels_pt.size(1)) # [B,T]
        diff = top_cand_scores - label_scores
        #diff = diff.masked_fill(~label_pt_mask, 0)

        margin = 1
        raw_score = sgn.float() * diff - margin
        raw_score = raw_score.masked_fill(raw_score > 0, 0)
        loss = -raw_score

        loss = loss.masked_fill(~label_pt_mask, 0)

        if self.config.fpd.margin_sq:
            loss = loss ** 2

        if self.config.fpd.margin_loss_reweight:
            coeff = torch.where(fgt_label == 1, self.config.fpd.margin_loss_pos_weight, 1.0).unsqueeze(-1)
            loss = coeff * loss

        mean_loss = loss.sum() / label_pt_mask.long().sum().float()
        return mean_loss

    def pred_forget_logit_based_multi(self, input_ids_pt, input_ids_ocl, attention_mask_pt, attention_mask_ocl, labels_pt, labels_ocl,
                decoder_attention_mask_pt, decoder_attention_mask_ocl, ocl_update_logits, pt_logits_ss, pt_logits_idxs, priors=None,
                forget_label=None, **kwargs):

        vocab_size = ocl_update_logits.size(-1)
        if self.config.fpd.logit_loss_type == 'kl':
            # collected lastest PT logits
            pt_logits_after_ss = kwargs.get('pt_logits_after_ss')
            pt_logits_after_idxs = kwargs.get('pt_logits_after_idxs')
            pt_logits_after_full = torch.full((pt_logits_after_ss.size(0), pt_logits_after_ss.size(1), vocab_size), -100.0).to(pt_logits_after_ss.device)
            pt_logits_after_mask = torch.zeros(pt_logits_after_ss.size(0), pt_logits_after_ss.size(1), vocab_size, dtype=torch.bool).to(pt_logits_after_ss.device)
            pt_logits_after_full.scatter_(2, pt_logits_after_idxs, pt_logits_after_ss)  # [B,T1,V]
            pt_logits_after_mask.scatter_(2, pt_logits_after_idxs, 1)  # [B,T1,V]

        rep_a = self.get_reps(input_ids_pt, attention_mask_pt, labels_pt, decoder_attention_mask_pt, all_ts=True) # [B,T1,H] 123 -> 111122223333
        rep_b = self.get_reps(input_ids_ocl, attention_mask_ocl, labels_ocl, decoder_attention_mask_ocl, all_ts=True) # [B,T2,H]  _> 123412341234

        # cosine similarity or simply dot product?
        rep_a_xxyy = torch.repeat_interleave(rep_a, rep_b.size(1), dim=1) # [B,T1*T2, H]
        rep_b_xyxy = rep_b.repeat(1, rep_a.size(1), 1)

        rep_dists = self.get_rep_prod(rep_a_xxyy, rep_b_xyxy)
        rep_dists = rep_dists.view(rep_a.size(0), rep_a.size(1), rep_b.size(1)) # [B,T1,T2]

        rep_dists_mask = self.get_rep_dists_mask(decoder_attention_mask_pt, decoder_attention_mask_ocl)
        rep_dists = rep_dists * rep_dists_mask.float()

        pt_logits_full = torch.full((pt_logits_ss.size(0), pt_logits_ss.size(1), vocab_size), -100.0).to(pt_logits_ss.device)
        pt_logits_mask = torch.zeros(pt_logits_ss.size(0), pt_logits_ss.size(1), vocab_size, dtype=torch.bool).to(pt_logits_ss.device)

        pt_logits_full.scatter_(2, pt_logits_idxs, pt_logits_ss) # [B,T1,V]
        pt_logits_mask.scatter_(2, pt_logits_idxs, 1) # [B,T1,V]

        # ocl_update_logits: [B,T2,V]
        # \Theta: [T1*V,T2*V][T2*V] --> T1*V or [T1,T2][T2,V] --> [T1,V]

        # reps: B*T1*H, B*T2*H
        dec_sum = decoder_attention_mask_ocl.sum(-1)
        #pred_logits_update = torch.matmul(rep_dists / rep_dists.size(-1), ocl_update_logits) # [B,T1,V] # normalized by T2 # fixed 0118

        pred_logits_update = torch.matmul(rep_dists / dec_sum.view(-1,1,1), ocl_update_logits)

        if self.use_vocab_map:
            #vocab_map = self.vocab_map.get_vocab_map()
            #pred_logits_update = torch.matmul(pred_logits_update, vocab_map) / vocab_map.size(0)
            pred_logits_update = self.vocab_map(pred_logits_update)

        pred_logits = pred_logits_update + pt_logits_full  # [B,T1,V]
        pred_logits_masked = pred_logits.masked_fill_(~pt_logits_mask, -np.inf)

        loss, loss_vec = None, None
        if forget_label is not None:
            if self.config.fpd.logit_loss_type == 'kl':
                #if self.config.fpd.use_margin_variant:
                #    loss = self.margin_loss(pred_logits, pt_logits_mask, labels_pt, forget_label, ref_mask=pt_logits_after_mask)
                #else:
                loss = self.masked_kl_loss(pred_logits, pt_logits_after_full, pt_logits_mask, pt_logits_after_mask, labels_pt, pt_logits_full)
            elif self.config.fpd.logit_loss_type == 'margin':
                if self.config.fpd.margin_multi_ts:
                    loss = self.margin_loss_multi(pred_logits, pt_logits_mask, labels_pt, forget_label)
                else:
                    loss = self.margin_loss(pred_logits, pt_logits_mask, labels_pt, forget_label)
            else:
                ce = F.cross_entropy(pred_logits_masked.view(-1, pred_logits_masked.size(2)), labels_pt.view(-1),
                                     reduction='none')
                ce = (ce.view(pred_logits_masked.size(0),
                              pred_logits_masked.size(1)) * decoder_attention_mask_pt.float()).mean(-1) # [B]
                sgn = torch.where(forget_label == 1, -1, 1)
                loss_vec = ce * sgn
                loss = loss_vec.mean()
        return pred_logits_masked, loss, loss_vec


    def infer_pred_forget_with_reps_logit_multi_batched(self, ocl_reps, all_pt_reps, all_pt_logits, ocl_logits_change,
                                                ocl_dec_attn_mask, all_pt_dec_attn_masks, return_pred_logits=False,
                                                skip_pred=False):
        # [1,T2,H], [N,T1,H]
        #ocl_reps = ocl_reps.unsqueeze(0)

        rep_dists = []
        for ocl_ts in range(ocl_reps.size(1)):
            ocl_rep_t = ocl_reps[:, ocl_ts] # [1,H]
            all_pt_reps_flat = all_pt_reps.view(-1, ocl_reps.size(2))  # [NT1, H]
            #rep_dist_t = F.cosine_similarity(ocl_rep_t, all_pt_reps_flat) # [NT1]
            rep_dist_t = self.get_rep_prod_mat(ocl_rep_t, all_pt_reps_flat)[0]
            rep_dists.append(rep_dist_t)
        rep_dists = torch.stack(rep_dists) # [T2,NT1]
        rep_dists = rep_dists.transpose(0,1) # [NT1,T2]
        rep_dists = rep_dists.view(all_pt_reps.size(0), all_pt_reps.size(1), rep_dists.size(1)) # [N,T1,T2]

        rep_dists_mask = self.get_rep_dists_mask_1vn(all_pt_dec_attn_masks, ocl_dec_attn_mask)
        rep_dists = rep_dists * rep_dists_mask.float()

        ocl_logits_change = torch.from_numpy(ocl_logits_change).to(rep_dists.device)

        # ocl_logits_change: [T2,V]
        pt_ds_size = all_pt_reps.size(0)
        ts_idx = self.config.fpd.ts
        preds_forget = []
        vocab_size = ocl_logits_change.size(-1)
        all_pred_pt_logits_top = []
        all_pred_pt_logits_top_idxs = []

        vocab_map = None
        current_slice = 0
        slice_size = 64

        pred_logit_update_slice = None

        for pt_idx in range(pt_ds_size):
            if pt_idx >= current_slice:
                pred_logit_update_slice = torch.matmul(rep_dists[pt_idx:pt_idx+slice_size] / rep_dists.size(-1), ocl_logits_change)  # [N,T1,V]
                if self.use_vocab_map:
                    pred_logit_update_slice = self.vocab_map(pred_logit_update_slice)
                current_slice += slice_size

            pt_logit_scores, pt_logit_idxs = all_pt_logits['logits'][pt_idx]
            pt_logit_scores, pt_logit_idxs = torch.from_numpy(pt_logit_scores).to(rep_dists.device), \
                                            torch.from_numpy(pt_logit_idxs).to(rep_dists.device)

            #rep_dist_i = rep_dists[pt_idx] # [T1,T2]
            # ignore pad
            #rep_dist_i = rep_dist_i[:pt_logit_scores.size(0)]

            #dist_logger.info('pt_idx: {}, rep_dist_i at ts {}: {}'.format(pt_idx, ts_idx, rep_dist_i[ts_idx].cpu().numpy()))

            #pred_logit_update = torch.matmul(rep_dist_i / rep_dist_i.size(1), ocl_logits_change) # [T1,V]
            pred_logit_update = pred_logit_update_slice[pt_idx % slice_size][:pt_logit_scores.size(0)]

            pt_logits_full, pt_logits_mask = self.get_sparse_pt_logits_no_batch(pt_logit_scores, pt_logit_idxs, vocab_size=vocab_size)

            pred_pt_logits_after = pt_logits_full + pred_logit_update
            pred_pt_logits_after.masked_fill_(~pt_logits_mask, -np.inf)

            pred_pt_logits_top, pred_pt_logits_top_idxs = pred_pt_logits_after.topk(100, -1) # [T1,Vs]
            all_pred_pt_logits_top.append(pred_pt_logits_top.detach().cpu().numpy())
            all_pred_pt_logits_top_idxs.append(pred_pt_logits_top_idxs.detach().cpu().numpy())

            if not skip_pred:
                pred_pt_logits_t = pred_pt_logits_after[ts_idx]
                pred_pt_logits_t[0] = -np.inf
                _, max_idx = pred_pt_logits_t.max(-1)

                max_idx = max_idx.item()
                label_t = all_pt_logits['labels'][pt_idx][ts_idx]

                if self.config.fpd.compare_tokens:
                    max_idx_tok, label_t_tok = self.tokenizer.convert_ids_to_tokens([max_idx, label_t])
                    if max_idx_tok.lower() != label_t_tok.lower():
                        preds_forget.append(pt_idx)
                else:
                    if max_idx != label_t:
                        preds_forget.append(pt_idx)

        if return_pred_logits:
            return preds_forget, all_pred_pt_logits_top, all_pred_pt_logits_top_idxs
        else:
            return preds_forget

    def infer_pred_forget_with_reps_logit_single(self, ocl_reps, all_pt_reps, all_pt_logits, ocl_logits_change):
        rep_dist = F.cosine_similarity(ocl_reps.view(1,-1), all_pt_reps)  # []
        #rep_dist = self.get_rep_prod_mat(ocl_reps.view(1,-1), all_pt_reps)
        pt_ds_size = all_pt_reps.size(0)
        ts_idx = self.config.fpd.ts
        preds_forget = []
        for pt_idx in range(pt_ds_size):
            pt_logit_scores, pt_logit_idxs = all_pt_logits['logits'][pt_idx]
            pt_confused_scores = pt_logit_scores[ts_idx]
            pt_confused_idxs = pt_logit_idxs[ts_idx]

            ocl_logits_change_ss = ocl_logits_change[ts_idx, pt_confused_idxs]
            coeff = rep_dist[pt_idx].item()

            #dist_logger.info('pt_idx: {}, rep_dist_i at ts {}: {}'.format(pt_idx, ts_idx, rep_dist[pt_idx].cpu().numpy()))
            pred_after_pt_logits = pt_confused_scores + coeff * ocl_logits_change_ss

            label = all_pt_logits['labels'][pt_idx][ts_idx]
            max_idx = np.argmax(pred_after_pt_logits)

            # ignore 0
            if pt_confused_idxs[max_idx] == 0:
                pred_after_pt_logits[max_idx] = -100
                max_idx = np.argmax(pred_after_pt_logits)

                # print(pt_confused_idxs[max_idx], label)
            if self.config.fpd.compare_tokens:
                max_idx_tok, label_t_tok = self.tokenizer.convert_ids_to_tokens([pt_confused_idxs[max_idx], label])
                if max_idx_tok.lower() != label_t_tok.lower():
                    preds_forget.append(pt_idx)
            else:
                if pt_confused_idxs[max_idx] != label:
                    preds_forget.append(pt_idx)
            #if pt_confused_idxs[max_idx] != label:
            #    preds_forget.append(pt_idx)
        return preds_forget


    def infer_pred_forget_with_reps_logit_multi(self, ocl_reps, all_pt_reps, all_pt_logits, ocl_logits_change,
                                                ocl_dec_attn_mask, all_pt_dec_attn_masks, return_pred_logits=False,
                                                skip_pred=False):
        # [1,T2,H], [N,T1,H]
        #ocl_reps = ocl_reps.unsqueeze(0)

        rep_dists = []
        for ocl_ts in range(ocl_reps.size(1)):
            ocl_rep_t = ocl_reps[:, ocl_ts] # [1,H]
            all_pt_reps_flat = all_pt_reps.view(-1, ocl_reps.size(2))  # [NT1, H]
            #rep_dist_t = F.cosine_similarity(ocl_rep_t, all_pt_reps_flat) # [NT1]
            rep_dist_t = self.get_rep_prod_mat(ocl_rep_t, all_pt_reps_flat)[0]
            rep_dists.append(rep_dist_t)
        rep_dists = torch.stack(rep_dists) # [T2,NT1]
        rep_dists = rep_dists.transpose(0,1) # [NT1,T2]
        rep_dists = rep_dists.view(all_pt_reps.size(0), all_pt_reps.size(1), rep_dists.size(1)) # [N,T1,T2]

        rep_dists_mask = self.get_rep_dists_mask_1vn(all_pt_dec_attn_masks, ocl_dec_attn_mask)
        rep_dists = rep_dists * rep_dists_mask.float()

        ocl_logits_change = torch.from_numpy(ocl_logits_change).to(rep_dists.device)
        if len(ocl_logits_change) > self.config.max_output_length:
            print('Fixing over len ocl_logits_change')
            ocl_logits_change = ocl_logits_change[:self.config.max_output_length]


        # ocl_logits_change: [T2,V]
        pt_ds_size = all_pt_reps.size(0)
        ts_idx = self.config.fpd.ts
        preds_forget = []
        vocab_size = ocl_logits_change.size(-1)
        all_pred_pt_logits_top = []
        all_pred_pt_logits_top_idxs = []

        if self.config.fpd.norm_by_inverse:
            self_rep_dists = torch.matmul(ocl_reps[0], ocl_reps[0].transpose(0,1)) # [T2,T2]
            if self.config.fix_label_bos:
                self_rep_dists_ind = self_rep_dists[self.config.fpd.ts:, self.config.fpd.ts:]
                self_rep_dists_inv = torch.zeros_like(self_rep_dists)
                self_rep_dists_inv[self.config.fpd.ts:, self.config.fpd.ts:] = torch.inverse(self_rep_dists_ind)
            else:
                self_rep_dists_inv = torch.inverse(self_rep_dists)

        for pt_idx in range(pt_ds_size):
            pt_logit_scores, pt_logit_idxs = all_pt_logits['logits'][pt_idx]
            pt_logit_scores, pt_logit_idxs = torch.from_numpy(pt_logit_scores).to(rep_dists.device), \
                                            torch.from_numpy(pt_logit_idxs).to(rep_dists.device)

            rep_dist_i = rep_dists[pt_idx] # [T1,T2]
            # ignore pad

            rep_dist_i = rep_dist_i[:pt_logit_scores.size(0)]

            if self.config.fpd.norm_by_inverse:
                pred_logit_update = torch.matmul(torch.matmul(rep_dist_i, self_rep_dists_inv), ocl_logits_change)
            else:
                if rep_dist_i.shape[-1] != ocl_logits_change.shape[0]:
                    print(rep_dist_i.shape, ocl_logits_change.shape)
                #pred_logit_update = torch.matmul(rep_dist_i / rep_dist_i.size(1), ocl_logits_change) # [T1,V] # fix here
                if self.config.fpd.method == 'logit_direct':
                    pred_logit_update = torch.matmul(rep_dist_i / 128, ocl_logits_change) # [T1,V]
                else:
                    pred_logit_update = torch.matmul(rep_dist_i / rep_dist_i.size(1), ocl_logits_change[:rep_dist_i.shape[1]]) # [T1,V] # fix here
            if self.use_vocab_map:
                pred_logit_update = self.vocab_map(pred_logit_update)
            pt_logits_full, pt_logits_mask = self.get_sparse_pt_logits_no_batch(pt_logit_scores, pt_logit_idxs, vocab_size=vocab_size)

            pred_pt_logits_after = pt_logits_full + pred_logit_update
            pred_pt_logits_after.masked_fill_(~pt_logits_mask, -np.inf)

            # special treatment for bart
            if 'bart0' in self.config.model_name.lower():
                pred_pt_logits_after[:,0] = -np.inf

            pred_pt_logits_top, pred_pt_logits_top_idxs = pred_pt_logits_after.topk(100, -1) # [T1,Vs]

            all_pred_pt_logits_top.append(pred_pt_logits_top.detach().cpu().numpy())
            all_pred_pt_logits_top_idxs.append(pred_pt_logits_top_idxs.detach().cpu().numpy())

            if not skip_pred:
                pred_pt_logits_t = pred_pt_logits_after[ts_idx]
                pred_pt_logits_t[0] = -np.inf
                _, max_idx = pred_pt_logits_t.max(-1)

                max_idx = max_idx.item()
                label_t = all_pt_logits['labels'][pt_idx][ts_idx]

                if self.config.fpd.compare_tokens:
                    max_idx_tok, label_t_tok = self.tokenizer.convert_ids_to_tokens([max_idx, label_t])
                    if max_idx_tok.lower() != label_t_tok.lower():
                        preds_forget.append(pt_idx)
                else:
                    if max_idx != label_t:
                        preds_forget.append(pt_idx)

        if return_pred_logits:
            return preds_forget, all_pred_pt_logits_top, all_pred_pt_logits_top_idxs
        else:
            return preds_forget

    def infer_pred_forget_with_reps_logit_multi_profile(self, ocl_reps, all_pt_reps, all_pt_logits, ocl_logits_change,
                                                ocl_dec_attn_mask, all_pt_dec_attn_masks, return_pred_logits=False,
                                                skip_pred=False):
        # [1,T2,H], [N,T1,H]
        #ocl_reps = ocl_reps.unsqueeze(0)

        rep_dists = []
        for ocl_ts in range(ocl_reps.size(1)):
            ocl_rep_t = ocl_reps[:, ocl_ts] # [1,H]
            all_pt_reps_flat = all_pt_reps.view(-1, ocl_reps.size(2))  # [NT1, H]
            #rep_dist_t = F.cosine_similarity(ocl_rep_t, all_pt_reps_flat) # [NT1]
            rep_dist_t = self.get_rep_prod_mat(ocl_rep_t, all_pt_reps_flat)[0]
            rep_dists.append(rep_dist_t)
        rep_dists = torch.stack(rep_dists) # [T2,NT1]
        rep_dists = rep_dists.transpose(0,1) # [NT1,T2]
        rep_dists = rep_dists.view(all_pt_reps.size(0), all_pt_reps.size(1), rep_dists.size(1)) # [N,T1,T2]

        rep_dists_mask = self.get_rep_dists_mask_1vn(all_pt_dec_attn_masks, ocl_dec_attn_mask)
        rep_dists = rep_dists * rep_dists_mask.float()

        ocl_logits_change = torch.from_numpy(ocl_logits_change).to(rep_dists.device)
        if len(ocl_logits_change) > self.config.max_output_length:
            print('Fixing over len ocl_logits_change')
            ocl_logits_change = ocl_logits_change[:self.config.max_output_length]


        # ocl_logits_change: [T2,V]
        pt_ds_size = all_pt_reps.size(0)
        ts_idx = self.config.fpd.ts
        preds_forget = []
        vocab_size = ocl_logits_change.size(-1)
        all_pred_pt_logits_top = []
        all_pred_pt_logits_top_idxs = []

        if self.config.fpd.norm_by_inverse:
            self_rep_dists = torch.matmul(ocl_reps[0], ocl_reps[0].transpose(0,1)) # [T2,T2]
            if self.config.fix_label_bos:
                self_rep_dists_ind = self_rep_dists[self.config.fpd.ts:, self.config.fpd.ts:]
                self_rep_dists_inv = torch.zeros_like(self_rep_dists)
                self_rep_dists_inv[self.config.fpd.ts:, self.config.fpd.ts:] = torch.inverse(self_rep_dists_ind)
            else:
                self_rep_dists_inv = torch.inverse(self_rep_dists)

        flops = []
        dist_n = []
        inner_module = LogitFpdForwardModule()

        for pt_idx in range(pt_ds_size):
            pt_logit_scores, pt_logit_idxs = all_pt_logits['logits'][pt_idx]
            pt_logit_scores, pt_logit_idxs = torch.from_numpy(pt_logit_scores).to(rep_dists.device), \
                                            torch.from_numpy(pt_logit_idxs).to(rep_dists.device)

            rep_dist_i = rep_dists[pt_idx] # [T1,T2]
            # ignore pad

            rep_dist_i = rep_dist_i[:pt_logit_scores.size(0)]

            if self.config.fpd.norm_by_inverse:
                pred_logit_update = torch.matmul(torch.matmul(rep_dist_i, self_rep_dists_inv), ocl_logits_change)
            else:
                if rep_dist_i.shape[-1] != ocl_logits_change.shape[0]:
                    print(rep_dist_i.shape, ocl_logits_change.shape)
                pred_logit_update = torch.matmul(rep_dist_i / rep_dist_i.size(1), ocl_logits_change) # [T1,V]

            flop = count_flops(inner_module, (rep_dist_i, ocl_logits_change),' inner')
            flops.append(flop)
            dist_n.append(rep_dist_i.size(0) * rep_dist_i.size(1))

            if self.use_vocab_map:
                pred_logit_update = self.vocab_map(pred_logit_update)
            pt_logits_full, pt_logits_mask = self.get_sparse_pt_logits_no_batch(pt_logit_scores, pt_logit_idxs, vocab_size=vocab_size)

            pred_pt_logits_after = pt_logits_full + pred_logit_update
            pred_pt_logits_after.masked_fill_(~pt_logits_mask, -np.inf)

            # special treatment for bart
            if 'bart0' in self.config.model_name.lower():
                pred_pt_logits_after[:,0] = -np.inf

            pred_pt_logits_top, pred_pt_logits_top_idxs = pred_pt_logits_after.topk(100, -1) # [T1,Vs]

            all_pred_pt_logits_top.append(pred_pt_logits_top.detach().cpu().numpy())
            all_pred_pt_logits_top_idxs.append(pred_pt_logits_top_idxs.detach().cpu().numpy())

            if not skip_pred:
                pred_pt_logits_t = pred_pt_logits_after[ts_idx]
                pred_pt_logits_t[0] = -np.inf
                _, max_idx = pred_pt_logits_t.max(-1)

                max_idx = max_idx.item()
                label_t = all_pt_logits['labels'][pt_idx][ts_idx]

                if self.config.fpd.compare_tokens:
                    max_idx_tok, label_t_tok = self.tokenizer.convert_ids_to_tokens([max_idx, label_t])
                    if max_idx_tok.lower() != label_t_tok.lower():
                        preds_forget.append(pt_idx)
                else:
                    if max_idx != label_t:
                        preds_forget.append(pt_idx)

        return preds_forget, flops, dist_n

    def mask_pad_in_labels(self, labels):
        ret = labels.masked_fill(labels==self.tokenizer.pad_token_id, -100)
        return ret

    def clean_batch(self, batch):
        batch['input_ids_pt'], batch['attention_mask_pt'] = trim_batch(batch['input_ids_pt'],
                                                                 self.tokenizer.pad_token_id, batch['attention_mask_pt'])
        batch['input_ids_ocl'], batch['attention_mask_ocl'] = trim_batch(batch['input_ids_ocl'],
                                                                 self.tokenizer.pad_token_id, batch['attention_mask_ocl'])
        batch['labels_pt'], batch['decoder_attention_mask_pt'] = trim_batch(batch['labels_pt'],
                                                                 self.tokenizer.pad_token_id, batch['decoder_attention_mask_pt'])
        batch['labels_ocl'], batch['decoder_attention_mask_ocl'] = trim_batch(batch['labels_ocl'],
                                                                   self.tokenizer.pad_token_id, batch['decoder_attention_mask_ocl'])

        batch['labels_ocl'] = self.mask_pad_in_labels(batch['labels_ocl'])
        batch['labels_pt'] = self.mask_pad_in_labels(batch['labels_pt'])

        batch.pop('input_ids')
        batch.pop('attention_mask')

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda()

        return batch

    def clean_batch_for_rep(self, batch):
        #batch['labels'] = self.mask_pad_in_labels(batch['labels'])
        batch['input_ids'], batch['attention_mask'] = trim_batch(batch['input_ids'], self.tokenizer.pad_token_id, batch['attention_mask'])
        batch['labels'], batch['decoder_attention_mask'] = trim_batch(batch['labels'], self.tokenizer.pad_token_id, batch['_decoder_attention_mask'])
        batch['labels'] = self.mask_pad_in_labels(batch['labels'])
        batch = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']}

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda()

        return batch

    def get_sparse_pt_logits_no_batch(self, pt_logits_ss, pt_logits_idxs, vocab_size):
        pt_logits_full = torch.full((pt_logits_ss.size(0), vocab_size), -100.0).to(pt_logits_ss.device)
        pt_logits_mask = torch.zeros(pt_logits_ss.size(0), vocab_size, dtype=torch.bool).to(pt_logits_ss.device)

        pt_logits_full.scatter_(1, pt_logits_idxs, pt_logits_ss) # [B,T,V]
        pt_logits_mask.scatter_(1, pt_logits_idxs, 1)
        return pt_logits_full, pt_logits_mask

    def pad_and_cat(self, reps_l):
        max_ts = max([x.size(1) for x in reps_l])
        padded_reps = []
        for rep in reps_l:
            if rep.size(1) < max_ts:
                pad = torch.zeros(rep.size(0), max_ts - rep.size(1), rep.size(2), dtype=rep.dtype).to(rep.device)
                p_rep = torch.cat([rep, pad], 1)
                padded_reps.append(p_rep)
            else:
                padded_reps.append(rep)
        padded_reps = torch.cat(padded_reps,0)
        return padded_reps

    def pad_and_cat_attn(self, attns_l):
        max_ts = max([x.size(1) for x in attns_l])
        padded_reps = []
        for rep in attns_l:
            if rep.size(1) < max_ts:
                pad = torch.zeros(rep.size(0), max_ts - rep.size(1), dtype=rep.dtype).to(rep.device)
                p_rep = torch.cat([rep, pad], 1)
                padded_reps.append(p_rep)
            else:
                padded_reps.append(rep)
        padded_reps = torch.cat(padded_reps,0)
        return padded_reps

    def convert_priors_to_bias(self, priors):
        if self.config.fpd.prior == 'odd':
            bias = torch.log(priors)
        else:
            raise NotImplementedError
        return bias

    def add_bias_to_logit_if_needed(self, logit, priors):
        if self.config.fpd.prior == 'odd':
            bias = self.convert_priors_to_bias(priors)
            logit = logit + bias
        return logit

    def get_rep_dists_mask(self, attn_mask_a, attn_mask_b):
        # [B,T1], [B,T2] -> res[b, T1, T2]
        mask = torch.ones(attn_mask_a.size(0), attn_mask_a.size(1), attn_mask_b.size(1)).to(attn_mask_a.device) # [B,T1,T2]
        for b in range(attn_mask_a.size(0)):
            mask[b, ~(attn_mask_a[b].bool())] = 0
            mask[b, :, ~(attn_mask_b[b].bool())] = 0
        return mask

    def get_rep_dists_mask_1vn(self, pt_dec_attn_mask, ocl_dec_attn_mask):
        #  [N,T1], [1,T2] -> res[N,T1,T2]
        assert ocl_dec_attn_mask.size(0) == 1
        mask = torch.ones(pt_dec_attn_mask.size(0), pt_dec_attn_mask.size(1), ocl_dec_attn_mask.size(1)).to(pt_dec_attn_mask.device)
        mask[~(pt_dec_attn_mask.bool())] = 0
        mask[:,:,~(ocl_dec_attn_mask[0].bool())] = 0
        return mask

class ForgetPredictionModelForCausualLM(ForgettingPredictionModel):
    def __init__(self, config, tokenizer, helper, init_model=True):
        super().__init__(config, tokenizer, helper, init_model=False)
        self.is_sent_encoder = 'MiniLM' in config.fpd.model_name
        if init_model:
            if self.is_sent_encoder:
                self.lm = AutoModel.from_pretrained(config.fpd.model_name, trust_remote_code=True)
            else:
                self.lm = AutoModelForCausalLM.from_pretrained(config.fpd.model_name, trust_remote_code=True)
            if self.use_head:
                self.head = ContrastiveHead(getattr(self.lm.config, 'd_model', self.lm.config.hidden_size), config.fpd.output_dim)

    def get_reps(self, input_ids, all_ts=False, **kwargs):
        input_ids = input_ids.cuda()
        if self.is_sent_encoder:
            outputs = self.lm(input_ids=input_ids)
            hidden = outputs[0]
        else:
            outputs = self.lm(input_ids=input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
        input_len = (input_ids!= self.tokenizer.pad_token_id).sum(-1)
        if all_ts:
            raw_rep = hidden # [B,T,H]
        else:
            raw_rep = hidden[torch.arange(hidden.size(0)), input_len - 1,:]

        if self.config.fpd.freeze_lm:
            raw_rep = raw_rep.detach()

        if self.use_head:
            rep = self.head(raw_rep)
        else:
            rep = raw_rep

        if self.normalize:
            rep = F.normalize(rep, dim=-1)

        return rep

    def pred_forget_pairwise_mse(self, input_ids_pt, input_ids_ocl, forget_label=None, **kwargs):
        rep_a = self.get_reps(input_ids_pt)
        rep_b = self.get_reps(input_ids_ocl)
        score = self.get_rep_prod(rep_a, rep_b)
        #print(forget_label, prob)

        loss = None
        if forget_label is not None:
            loss = F.mse_loss(score, forget_label.float())
        return score, loss

    def pred_forget_pairwise_ce(self, input_ids_pt, input_ids_ocl, forget_label=None, **kwargs):
        rep_a = self.get_reps(input_ids_pt)
        rep_b = self.get_reps(input_ids_ocl)
        logit = self.get_rep_prod(rep_a, rep_b)

        prob = F.sigmoid(logit)
        loss = None
        weights = torch.where(forget_label == 1, self.config.fpd.ce_loss_pos_weight, 1.)

        if forget_label is not None:
            loss = F.binary_cross_entropy(prob, forget_label.float(), weight=weights)
        return prob, loss


    def pred_forget_with_reps_score(self, all_ocl_reps, all_pt_reps, thres=0.):
        scores = self.get_rep_prod_mat(all_ocl_reps, all_pt_reps)
        preds = (scores > thres).long()
        #print(prob_grid, preds)
        return scores, preds

    def batch_to_cuda(self, batch):
        batch_ = {k:v for k,v in batch.items()}
        for k,v in batch_.items():
            if torch.is_tensor(v):
                batch_[k] = v.cuda()
        return batch_
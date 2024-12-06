def infer_pred_forget_with_reps_logit_multi_(self, ocl_reps, all_pt_reps, all_pt_logits, ocl_logits_change,
                                            ocl_dec_attn_mask, all_pt_dec_attn_masks, return_pred_logits=False,
                                            skip_pred=False):
    # [1,T2,H], [N,T1,H]
    # ocl_reps = ocl_reps.unsqueeze(0)

    rep_dists = []
    for ocl_ts in range(ocl_reps.size(1)):
        ocl_rep_t = ocl_reps[:, ocl_ts]  # [1,H]
        all_pt_reps_flat = all_pt_reps.view(-1, ocl_reps.size(2))  # [NT1, H]
        # rep_dist_t = F.cosine_similarity(ocl_rep_t, all_pt_reps_flat) # [NT1]
        rep_dist_t = self.get_rep_prod_mat(ocl_rep_t, all_pt_reps_flat)[0]
        rep_dists.append(rep_dist_t)
    rep_dists = torch.stack(rep_dists)  # [T2,NT1]
    rep_dists = rep_dists.transpose(0, 1)  # [NT1,T2]
    rep_dists = rep_dists.view(all_pt_reps.size(0), all_pt_reps.size(1), rep_dists.size(1))  # [N,T1,T2]

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
    if self.use_vocab_map:
        vocab_map = self.vocab_map.get_vocab_map()

    for pt_idx in range(pt_ds_size):
        pt_logit_scores, pt_logit_idxs = all_pt_logits['logits'][pt_idx]
        pt_logit_scores, pt_logit_idxs = torch.from_numpy(pt_logit_scores).to(rep_dists.device), \
                                         torch.from_numpy(pt_logit_idxs).to(rep_dists.device)

        rep_dist_i = rep_dists[pt_idx]  # [T1,T2]
        # ignore pad
        rep_dist_i = rep_dist_i[:pt_logit_scores.size(0)]

        # dist_logger.info('pt_idx: {}, rep_dist_i at ts {}: {}'.format(pt_idx, ts_idx, rep_dist_i[ts_idx].cpu().numpy()))

        pred_logit_update = torch.matmul(rep_dist_i / rep_dist_i.size(1), ocl_logits_change)  # [T1,V]

        if self.use_vocab_map:
            pred_logit_update = torch.matmul(pred_logit_update, vocab_map) / vocab_map.size(0)

        pt_logits_full, pt_logits_mask = self.get_sparse_pt_logits_no_batch(pt_logit_scores, pt_logit_idxs,
                                                                            vocab_size=vocab_size)

        pred_pt_logits_after = pt_logits_full + pred_logit_update
        pred_pt_logits_after.masked_fill_(~pt_logits_mask, -np.inf)

        pred_pt_logits_top, pred_pt_logits_top_idxs = pred_pt_logits_after.topk(100, -1)  # [T1,Vs]
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


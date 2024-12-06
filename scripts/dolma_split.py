import os
import argparse
from data_utils.lm import DolmaDataset
from utils.config import load_configs
import random as _random
import numpy as np
from more_itertools import chunked
import pickle
from transformers import AutoTokenizer
import multiprocessing
import gzip
import json


def pad(arr):
    if len(arr) != config.max_input_length:
        new_arr = np.concatenate([arr, np.full(config.max_input_length - len(arr),
                                              pad_token_id).astype(arr.dtype)])
    else:
        new_arr = arr
    return new_arr

def process_single(ds_idx, files):
    res = []
    with open(os.path.join(raw_text_dir, files[ds_idx]), 'rb') as f:
        ds_arr = np.fromfile(f, np.int64)
    idx_arr = np.arange(len(ds_arr))
    eot_pos = idx_arr[ds_arr == eot_id]
    print('{} complete text in {}'.format(len(eot_pos), ds_idx))

    eot_pos_shift = np.zeros_like(eot_pos)
    eot_pos_shift[1:] = eot_pos[:-1]
    lens = eot_pos - eot_pos_shift

    for sent_idx in range(len(eot_pos)):
        if sent_idx % 1000 == 0:
            print('{} done {}\r'.format(ds_idx, sent_idx))
        sent = ds_arr[eot_pos_shift[sent_idx]:eot_pos[sent_idx] + 1]  # including end of text
        for chunk in chunked(sent, args.chunk_size):
            res.append([tokenizer.decode(chunk), sent_idx])

    with open(os.path.join(args.output_path, 'chunked_{}.pkl'.format(ds_idx)), 'wb') as wf:
        pickle.dump(res, wf)

def tokenize_and_process_single_dolma(ds_idx, files):
    res = []
    with gzip.open(os.path.join(raw_text_dir, files[ds_idx]),'rb') as f:
        lines = f.readlines()

    batch_texts = []

    for lidx, line in enumerate(lines):
        data = json.loads(line)
        text = data['text'] + '<|endoftext|>'
        batch_texts.append(text)
        #if lidx == 10:
        #    break
    print('tokenizing {}\r'.format(ds_idx))
    encoded = tokenizer.batch_encode_plus(batch_texts)
    token_ids = encoded['input_ids']

    for sent_idx in range(len(token_ids)):
        if sent_idx % 1000 == 0:
            print('{} done {}\r'.format(ds_idx, sent_idx))
        sent = token_ids[sent_idx]
        for chunk in chunked(sent, args.chunk_size):
            res.append([tokenizer.decode(chunk), sent_idx])

    with open(os.path.join(args.output_path, 'chunked_{}.pkl'.format(ds_idx)), 'wb') as wf:
        pickle.dump(res, wf)

if __name__ == '__main__':
    random = _random.Random(0)
    pad_token_id = 1
    #raw_dir = '/home/xsjin/cl-analysis/data/dolma_v1_6-sample_tok_mm'
    raw_text_dir = '/home/xsjin/cl-analysis/data/dolma_v1_6-sample'
    eot_id = 50279

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-Instruct", trust_remote_code=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path')
    parser.add_argument('--chunk_size', type=int, default=2047)
    parser.add_argument('--config_files',nargs='*')
    parser.add_argument('--frac', type=float, default=0.01)
    parser.add_argument('--natural', action='store_true')
    args = parser.parse_args()

    if args.config_files:
        config = load_configs(*args.config_files)
    else:
        config = None
    ds_num = len(os.listdir(raw_text_dir))

    all_data = []

    if args.natural:
        files = os.listdir(raw_text_dir)
        #for ds_idx in range(len(files)):
        #    tokenize_and_process_single_dolma(ds_idx, files)
        with multiprocessing.Pool(processes=32) as pool:
            procs = {ds_idx: pool.apply_async(tokenize_and_process_single_dolma, (ds_idx,files)) for ds_idx in range(ds_num)}
            results = {seed: proc.get() for seed, proc in procs.items()}

    else:
        for ds_idx in range(ds_num):
            ds = DolmaDataset(config, None)
            ds.load_all_from_dir(raw_dir, ds_idx, ds_idx + 1)

            # sample_num
            num = int(len(ds) * args.frac)
            idxs = random.sample(range(len(ds)), num)

            print('Sampling {} items for shard {}'.format(num, ds_idx))
            items = [ds[x]['input_ids'] for x in idxs]

            pad_items = [pad(x) for x in items]
            all_data.append(np.stack(pad_items))

        all_data = np.concatenate(all_data)
        print('Total len {}'.format(all_data.shape))
        with open(args.output_path,'wb') as wf:
            all_data.tofile(wf)

import os
import numpy as np
from transformers import AutoTokenizer
import argparse
import gzip
import json

def do_tokenize(base_dir):
    #base_dir = 'data/paloma'
    output_base_dir = base_dir + '_tok'

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)

    for dir_name, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith('jsonl.gz'):
                path = os.path.join(dir_name, filename)
                output_dir = dir_name.replace(base_dir, output_base_dir)

                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename[:-len('.jsonl.gz')] + '.npy')

                print(path)

                if not os.path.exists(output_path):
                    with gzip.open(path) as f:
                        data = json.loads(f.readlines()[0])
                        text = data['text'] + '<|endoftext|>'

                        token_ids = np.array(tokenizer.encode(text), dtype=np.int32)
                        np.save(output_path, token_ids)

                    print(output_path)

def do_tokenize_multi(base_dir):
    #base_dir = 'data/paloma'
    output_base_dir = base_dir + '_tok'

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)

    for dir_name, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith('json.gz'):
                path = os.path.join(dir_name, filename)
                output_dir = dir_name.replace(base_dir, output_base_dir)

                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename[:-len('.json.gz')] + '.npy')

                print(path)

                all_token_ids = []
                if not os.path.exists(output_path):
                    with gzip.open(path) as f:
                        lines = f.readlines()

                    batch_texts = []
                    for line in lines:

                        data = json.loads(line)
                        text = data['text'] + '<|endoftext|>'
                        batch_texts.append(text)

                    encoded = tokenizer.batch_encode_plus(batch_texts)

                    token_ids = np.concatenate(encoded['input_ids'])

                    np.save(output_path, token_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenize", action='store_true')
    parser.add_argument("--tokenize_multi", action='store_true')
    parser.add_argument("--base_dir")

    args = parser.parse_args()

    if args.tokenize:
        do_tokenize(args.base_dir)
    elif args.tokenize_multi:
        do_tokenize_multi(args.base_dir)
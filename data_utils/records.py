import os
import pickle

def get_key(entry):
    return entry['update_premise'] + '#' + entry['update_hypothesis']

def load_record_and_reorg_file(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    robj = {}
    for k, v in obj.items():
        if k != 'base_info':
            key = get_key(v)
            robj[key] = v
    return obj, robj
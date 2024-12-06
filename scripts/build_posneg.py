from data_utils.records import load_record_and_reorg_file
import random
import json
import argparse
import os

def construct_pairs_aff_naff(pt_examples, key2ocl):
    posneg_examples = []
    id2pt_example = {}
    # get idx of pt examples that are visible to this partition
    total_idxs = [x['idx'] for x in pt_examples]
    # print(total_idxs)

    for example in pt_examples:
        id2pt_example[example['idx']] = example
    for key, ocl_entry in key2ocl.items():
        ocl_premise = ocl_entry['update_premise']
        ocl_hypo = ocl_entry['update_hypothesis']
        ocl_label = ocl_entry['update_label']
        ocl_pred = ocl_entry['pred_label']

        # get index of unaffected examples
        aff_example_idxs = []
        for pt_entry in ocl_entry['new_err']:
            idx = pt_entry['idx']
            if idx in total_idxs:
                aff_example_idxs.append(pt_entry['idx'])
        naff_example_idxs = list(set(total_idxs).difference(aff_example_idxs))

        # for every single affected example, randomly sample another unaffected example
        for pt_entry in ocl_entry['new_err']:
            pt_example_idx = pt_entry['idx']
            if pt_example_idx not in aff_example_idxs:
                continue
            pt_premise = pt_entry['premise']
            pt_hypo = pt_entry['hypothesis']
            pt_label = pt_entry['label']

            neg_idx = random.choice(naff_example_idxs)
            neg_premise = id2pt_example[neg_idx]['premise']
            neg_hypo = id2pt_example[neg_idx]['hypothesis']
            neg_label = id2pt_example[neg_idx]['label']

            posneg_examples.append({
                'ocl_premise': ocl_premise,
                'ocl_hypo': ocl_hypo,
                'ocl_label': ocl_label,
                # 'ocl_pred': ocl_pred,
                'pos_premise': pt_premise,
                'pos_hypo': pt_hypo,
                'pos_label': pt_label,
                'pos_example_idx': pt_example_idx,
                'neg_premise': neg_premise,
                'neg_hypo': neg_hypo,
                'neg_label': neg_label,
                'neg_example_idx': neg_idx
            })
    return posneg_examples


def add_id(pt_examples):
    for i, example in enumerate(pt_examples):
        example['idx'] = i

def save_splits_to_file(keys, key2ocl, pt_examples, split, dir_format='tmp/{split}.json'):
    save_file = dir_format.format(split=split)
    key2ocl = {k: key2ocl[k] for k in keys}
    posneg_examples = construct_pairs_aff_naff(pt_examples, key2ocl)
    with open(save_file, 'w') as wf:
        json.dump(posneg_examples,wf)

def main(args):
    error_pkl_file = args.error_pkl_file
    output_dir = os.path.dirname(error_pkl_file)
    raw_record, key2ocl = load_record_and_reorg_file(error_pkl_file)
    pt_examples = raw_record['base_info']['base_examples']
    print('Total {} base PT examples'.format(len(pt_examples)))
    add_id(pt_examples)
    print('Total {} OCL examples'.format(len(key2ocl)))

    keys = [_ for _ in key2ocl]
    train_keys, dev_keys, test_keys = keys[:int(len(keys) * 0.6)], \
                                      keys[int(len(keys) * 0.6):int(len(keys) * 0.8)], \
                                      keys[int(len(keys) * 0.8):]
    train_pt_examples, dev_pt_examples, test_pt_examples = pt_examples[:int(len(pt_examples) * 0.6)], \
                                                        pt_examples[int(len(pt_examples) * 0.6):int(len(pt_examples) * 0.8)], \
                                                        pt_examples[int(len(pt_examples) * 0.8):]

    save_splits_to_file(train_keys, key2ocl, train_pt_examples, 'train', output_dir + '/postneg_train' + args.postfix)
    save_splits_to_file(dev_keys, key2ocl, dev_pt_examples, 'dev', output_dir + '/postneg_dev' + args.postfix)
    save_splits_to_file(test_keys, key2ocl, test_pt_examples, 'test', output_dir + '/postneg_test' + args.postfix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('error_pkl_file')
    parser.add_argument('postfix')
    args = parser.parse_args()

    main(args)
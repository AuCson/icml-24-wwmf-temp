import pickle
import csv
import os
import argparse

def split_ocl_logits_change_log(obj, thres1):
    train_obj, dev_obj = {}, {}
    for k, v in obj.items():
        #print(type(v))
        train_obj[k] = obj[k][:thres1]
        dev_obj[k] = obj[k][thres1:]
    return train_obj, dev_obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--thres', default=0.6)
    args = parser.parse_args()

    with open(os.path.join(args.dir, 'ocl_error_ds.csv')) as f:
        reader = csv.reader(f)
        rows = [_ for _ in reader]

    with open(os.path.join(args.dir, 'ocl_log.pkl'),'rb') as f:
        obj = pickle.load(f)
    ocl_idxs = sorted([_ for _ in obj.keys()])

    print(len(rows), len(obj))
    #assert len(obj) == len(rows)

    if len(rows) > len(obj):
        rows = rows[:len(obj)]
    thres1 = int(args.thres * len(rows))
    print(thres1)
    train_rows, dev_rows = rows[:thres1], rows[thres1:]

    #exit(-1)
    train_idxs, dev_idxs = ocl_idxs[:thres1], ocl_idxs[thres1:]
    train_obj, dev_obj = {k:v for k,v in obj.items() if k in train_idxs}, {k:v for k,v in obj.items() if k in dev_idxs}

    train_out_dir, dev_out_dir = os.path.join(args.dir,'fpd_train'), os.path.join(args.dir,'fpd_dev')
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(dev_out_dir, exist_ok=True)

    with open(os.path.join(train_out_dir,'ocl_error_ds.csv'),'w') as wf:
        writer = csv.writer(wf)
        writer.writerows(train_rows)

    with open(os.path.join(dev_out_dir,'ocl_error_ds.csv'),'w') as wf:
        writer = csv.writer(wf)
        writer.writerows(dev_rows)

    with open(os.path.join(train_out_dir,'ocl_log.pkl'),'wb') as wf:
        pickle.dump(train_obj, wf)

    with open(os.path.join(dev_out_dir,'ocl_log.pkl'),'wb') as wf:
        pickle.dump(dev_obj, wf)

    logit_change_file = os.path.join(args.dir, 'ocl_error_ds_change_v2_logit_change_eval.pkl')
    if os.path.exists(logit_change_file):
        with open(logit_change_file, 'rb') as f:
            obj = pickle.load(f)
        train_obj, dev_obj = split_ocl_logits_change_log(obj, thres1)

        with open(os.path.join(train_out_dir, 'ocl_error_ds_change_v2_logit_change_eval.pkl'), 'wb') as wf:
            pickle.dump(train_obj, wf)

        with open(os.path.join(dev_out_dir, 'ocl_error_ds_change_v2_logit_change_eval.pkl'), 'wb') as wf:
            pickle.dump(dev_obj, wf)

    # with open(os.path.join(args.dir, 'pt_logits_update_eval.pkl'),'rb') as f:
    #     obj = pickle.load(f)
    #     train_obj, dev_obj = obj[:thres1], obj[thres1:]
    #
    # with open(os.path.join(train_out_dir,'pt_logits_update_eval.s.pkl'),'wb') as wf:
    #     pickle.dump(train_obj, wf)
    #
    # with open(os.path.join(dev_out_dir, 'pt_logits_update_eval.s.pkl'), 'wb') as wf:
    #     pickle.dump(dev_obj, wf)


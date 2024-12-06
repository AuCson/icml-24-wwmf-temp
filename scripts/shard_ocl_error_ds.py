import csv
import argparse
import random as _random
import os
import pickle

random = _random.Random(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('total',type=int)
    args = parser.parse_args()

    meta_file = os.path.join(args.dir, 'shards_{}.pkl'.format(args.total))

    if os.path.exists(meta_file):
        raise FileExistsError

    with open(os.path.join(args.dir, 'ocl_error_ds.csv')) as f:
        rows = [_ for _ in csv.reader(f)]

    idxs = random.sample(range(len(rows)), len(rows))

    shards = {}
    shard_size = len(rows) // args.total
    for sid in range(args.total):
        start = shard_size * sid
        stop = shard_size * (sid + 1) if sid != args.total - 1 else args.total
        shards[sid] = idxs[start:stop]

        with open(os.path.join(args.dir, 'ocl_error_ds_{}_{}.csv'.format(sid, args.total)),'w') as wf:
            writer = csv.writer(wf)
            writer.writerows([rows[x] for x in shards[sid]])

    with open(meta_file,'wb') as wf:
        pickle.dump(shards, wf)


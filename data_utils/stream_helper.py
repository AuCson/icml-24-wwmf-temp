import random
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Subset, SubsetRandomSampler

class StreamHelper:
    def __init__(self, config, ocl_error_ds, collator):
        self.config = config
        self.ds = ocl_error_ds
        self.collator = collator
        self.random = random.Random(config.stream.seed)
        if ocl_error_ds is not None:
            generator = torch.Generator()
            generator.manual_seed(self.config.stream.seed)
            self.dataloader = DataLoader(self.ds, config.stream.bs, shuffle=True, collate_fn=collator, generator=generator)
            self.collator = collator
            self.iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            data = next(self.iter)
        return data

    def generate_single_stream(self, n_batch=None, n_repeat=None, seed=None):
        if n_batch is None:
            n_batch = self.config.stream.n_batch_per_stream
        #if n_repeat is None:
        #    n_repeat = self.config.jf
        if seed is None:
            print('Warning: using default random seed 0 for stream generation')
            seed = 0
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.dataloader = DataLoader(self.ds, self.config.stream.bs, shuffle=True, collate_fn=self.collator, generator=generator)
        stream = []

        if n_batch == -1: # iterate over the stream for once
            n_batch = len(self.dataloader)
        for n in range(n_batch):
            batch = self.get_next_batch()
            #for r in range(n_repeat):
            stream.append(batch)

        return stream

    def generate_random_subset_loader(self, seed, n_example):
        indices = [_ for _ in range(len(self.ds))]
        if n_example >= len(self.ds):
            raise ValueError('{} >= {}'.format(n_example, len(indices)))
        sampled_idxs = random.Random(seed).sample(indices, n_example)
        subset = Subset(self.ds, sampled_idxs)
        subset_dataloader = DataLoader(subset, self.config.stream.bs, shuffle=True, collate_fn=self.collator)
        return subset_dataloader, subset, sampled_idxs

    def get_instance_dataloader(self):
        dataloader = DataLoader(self.ds, 1, shuffle=False, collate_fn=self.collator)
        return dataloader

    def get_task_dataloader(self):
        idx_by_task = {}
        for i, x in enumerate(self.ds):
            task = x['task_name']
            if task not in idx_by_task:
                idx_by_task[task] = []
            idx_by_task[task].append(i)
        idx_by_ds = {
            task: Subset(self.ds, idxs) for task, idxs in idx_by_task.items()
        }

        tasks = [_ for _ in idx_by_ds.keys()]
        dss = [idx_by_ds[task] for task in tasks]
        loaders = [DataLoader(ds, self.config.stream.bs, shuffle=True, collate_fn=self.collator) for ds in dss]
        return tasks, loaders

    def decide_n_example(self):
        return int(self.config.subset.n_example_ratio * len(self.ds))


    def get_stream_meta(self, stream):
        ret = {}
        if 'case_id' in stream[0]:
            ret['case_id'] = [batch['case_id'] for batch in stream]
        if 'example_id' in stream[0]:
            ret['example_id'] = [batch['example_id'] for batch in stream]
        return ret

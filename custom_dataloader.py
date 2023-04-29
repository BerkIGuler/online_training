import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import RandomSampler
import math
from torchvision import datasets, transforms




class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, incoming_dataset, memory_dataset, sub_batch_sizes):
        self.incoming_dataset = incoming_dataset
        self.memory_dataset = memory_dataset
        self.batch_size = sum(sub_batch_sizes)
        self.sub_batch_sizes = sub_batch_sizes
        self.number_of_datasets = 2
        self.max_steps = self._get_max_steps()

    def _get_max_steps(self):

        return max(math.ceil(len(self.incoming_dataset) / self.sub_batch_sizes[0]), math.ceil(len(self.memory_dataset) / self.sub_batch_sizes[1]))

    def __len__(self):
        return self.max_steps * self.batch_size

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []

        sampler = RandomSampler(self.incoming_dataset)
        samplers_list.append(sampler)
        cur_sampler_iterator = sampler.__iter__()
        sampler_iterators.append(cur_sampler_iterator)

        sampler = RandomSampler(self.memory_dataset)
        samplers_list.append(sampler)
        cur_sampler_iterator = sampler.__iter__()
        sampler_iterators.append(cur_sampler_iterator)


        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.sub_batch_sizes
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.max_steps

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(epoch_samples):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab[i]):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)




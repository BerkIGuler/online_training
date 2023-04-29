from utils.datasets_onl import LoadImagesAndLabels
import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
from torch.utils.data.dataset import ConcatDataset
from custom_dataloader_original import BatchSchedulerSampler
from torch.utils.data import Subset
from utils.labeled_datasets import LabeledSubset, LabeledConcatDataset
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.datasets_onl import _RepeatSampler 
import pickle
from copy import deepcopy
#from pycocotools import mask as maskUtils
from torchvision.utils import save_image
from torchvision.ops import roi_pool, roi_align, ps_roi_pool, ps_roi_align

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str
from utils.torch_utils import torch_distributed_zero_first


def create_dataloader(incoming_path, memory_path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', nb_samples=100):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        incoming_samples = LoadImagesAndLabels(incoming_path, imgsz, batch_size//2,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)
                                    
    #print("Print1")
    #print(incoming_samples.labels)
    #print("Print1")
    print(len(incoming_samples))
    with torch_distributed_zero_first(rank):
        memory_data = LoadImagesAndLabels(memory_path, imgsz, batch_size//2,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)
    #total_index = list(range(len(memory_data)))
    get_index = random.sample(range(len(memory_data)), nb_samples)
    #dndndn = Subset(memory_data,get_index)
    #print(len(dndndn))
    memory_samples = LabeledSubset(memory_data,get_index)
    #memory_samples = memory_data[]
    #print(memory_samples)
    
    #print(len(memory_samples))
    batch_size = min(batch_size, len(memory_samples), len(incoming_samples))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    #sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataset = LabeledConcatDataset([incoming_samples, memory_samples])
    #print(len(dataset))
    sampler = BatchSchedulerSampler(dataset, batch_size =(batch_size//2 + batch_size//2) ,sub_batch_sizes=[batch_size//2,batch_size//2])
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()

    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

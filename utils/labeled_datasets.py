from torch.utils.data import Subset
from torch.utils.data.dataset import ConcatDataset
import numpy as np
from utils.datasets_onl import LoadImagesAndLabels
#class LabeledSubset(Subset, LoadImagesAndLabels):
#    def __init__(self, dataset, indices):
#        Subset.__init__(self, dataset, indices)
#        LoadImagesAndLabels.__init__(self, self.dataset.path, self.dataset.img_size, self.dataset.augment, self.dataset.hyp, self.dataset.rect, self.dataset.image_weights,
#                 self.dataset.stride)

class LabeledSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices
        self.labels = dataset.labels
        self.shapes = dataset.shapes
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class LabeledConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.labels = []
        self.shapes = []
        for dataset in datasets:
            self.labels += dataset.labels
            self.shapes.extend(dataset.shapes)
        self.labels = np.array(self.labels)
        self.shapes = np.array(self.shapes)

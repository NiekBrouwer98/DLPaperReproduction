import torch
import torchvision
from scipy.special import comb
import numpy as np
import pickle
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import Sampler

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class SourceSampler(Sampler):
    """"
    Sample batches
    """
    def __init__(self, data_source, batch_k=2, batch_size=32):
        self.data_source = data_source
        self.batch_k = batch_k
        self.num_samples = len(self.data_source)
        self.batch_size = batch_size
        print('number of data:', len(self.data_source))

        labels, num_samples = np.unique(self.data_source.labels, return_counts=True)
        self.max_samples = max(num_samples)
        self.min_samples = min(num_samples)
        self.labels = labels

        assert self.min_samples >= self.batch_k

    def __len__(self):
        # return self.num_samples * self.batch_size * 2
        iter_len = len(self.labels) * comb(self.min_samples, self.batch_k)
        iter_len = int(iter_len // self.batch_size)
        return iter_len

    def __iter__(self):
        for i in range(2):  # (self.__len__()):
            # sample both positive and negative labels
            pos_labels = np.random.choice(self.labels, int(self.batch_size / (2 * self.batch_k)), replace=False)
            print('positieve labels', pos_labels)
            neg_labels = np.random.choice(self.labels, int(self.batch_size / (2 * self.batch_k)), replace=False)
            print('negatieve labels', neg_labels)
            ret_idx = []
            for label in pos_labels:
                ret_idx.extend(np.random.choice(self.data_source.idx[label][label], 2, replace=False))
                # print('\t\tpositive label ', label, '\t', ret_idx[-2:]) 
            for label in neg_labels:
                neg_label = np.random.choice([l for l in self.labels if l != label], 1)[0]
                label_idx = np.random.choice(self.data_source.idx[label][label], 1)
                neg_label_idx = np.random.choice(self.data_source.idx[neg_label][neg_label], 1)
                ret_idx.extend([label_idx[0], neg_label_idx[0]])
            print('sampler:', (ret_idx))
            yield ret_idx


class MetricData(torch.utils.data.Dataset):
    """"
    Preprocessing the data
    """

    def __init__(self, data_root, anno_file, idx_file, return_fn=False):
        self.return_fn = return_fn
        if idx_file.endswith('pkl'):
            with open(idx_file, 'rb') as f:
                a = pickle.load(f)
                dictIDX = a.to_dict()
                self.idx = dict()
                # For loop to remove al the nan values and make a dictionary of the idx file
                for i in range(int(a.columns.values.tolist()[0]), int(a.columns.values.tolist()[99]) + 1):
                    b = dictIDX[i]
                    c = [b[k] for k in b]
                    newlist = [x for x in c if math.isnan(x) == False]
                    newlist = list(map(int, newlist))
                    self.idx[i] = {i: newlist}

        self.anno = pd.read_excel(anno_file)
        self._convert_labels()
        self.data_root = data_root

        # check if it is train or test set en do the right transform on the data  #torchvision.transforms.Lambda(self.pad), \
        if self.labels[0] == 1:
            self.transforms = transforms.Compose([transforms.Resize(size=256),
                                                  transforms.RandomCrop((224, 224)),
                                                  transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                  transforms.Normalize(mean=mean, std=std)])
        else:
            self.transforms = transforms.Compose([transforms.Resize(size=255),
                                                  transforms.CenterCrop((224, 224)),
                                                  transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                                  transforms.Normalize(mean=mean, std=std)])

    def pad(img, size_max=256):
        """
        Pads images to the specified size (height x width). 
        """

        pad_height = max(0, size_max - img.height)

        pad_width = max(0, size_max - img.width)

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        return transforms.functional.pad(
            img,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))

    def __getitem__(self, i):
        # print('__getitem__\t', i, i%16, '\tlabel:', self.labels[i])
        img = Image.open(os.path.join(self.data_root, self.fns[int(i - (self.labels[0] - 1) * 58.64)])).convert('RGB')
        img = self.transforms(img)
        return img if not self.return_fn else (img, self.labels[int(i - (self.labels[0] - 1) * 58.64)])

    def __len__(self):
        return self.anno.size

    def _convert_labels(self):
        labels, fns = [], []
        for i in range(0, self.anno.shape[0]):
            labels.append(int(self.anno.iat[i, 5]))
            fns.append(self.anno.iat[i, 0])
        self.labels = labels
        self.fns = fns

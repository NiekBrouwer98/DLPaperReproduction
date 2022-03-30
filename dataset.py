import pickle
import scipy.io as sio
import torch
import os
import numpy as np
from PIL import Image
from scipy.special import comb
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF

mean, std = [0.485, 0.456, 0.406], [1.0, 1.0, 1.0]

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/std[0], 1/std[1], 1/std[2] ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

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

    return TF.pad(
        img,
        (pad_left, pad_top, pad_right, pad_bottom),
        fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))

transformations = transforms.Compose([
   transforms.Resize(size=255, max_size=256),
   transforms.Lambda(pad),
   transforms.CenterCrop((224, 224)),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
    # loader=lambda x: Image.open(x).convert('RGB') todo: Add this one?
])

#Read images and label here
class ImageFolderWithName(datasets.ImageFolder):
    def __init__(self, return_fn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_fn = return_fn

    def __getitem__(self, i):
        img, label = super(ImageFolderWithName, self).__getitem__(i)
        if not self.return_fn:
            return img, label
        else:
            return img, label, self.imgs[i]

class MetricData(torch.utils.data.Dataset):
    def __init__(self, data_root, anno_file, idx_file, return_fn=False):
        self.return_fn = return_fn
        if idx_file.endswith('pkl'):
            with open(idx_file, 'rb') as f:
                self.idx = pickle.load(f)
        assert anno_file.endswith('mat')
        self.anno = sio.loadmat(anno_file)['annotations']
        self._convert_labels()
        self.data_root = data_root
        self.transforms = transformations

    def __len__(self):
        return self.anno.shape[1]
    
    def _convert_labels(self):
        labels, fns = [], []
        for i in range(self.anno.shape[1]):
            labels.append(self.anno[0, i][-2][0, 0])
            fns.append(self.anno[0, i][-1][0])
        self.labels = labels
        self.fns = fns

    @classmethod
    def tensor2img(cls, tensor):
        if type(tensor) != np.ndarray:
            tensor = tensor.cpu().numpy()

        if len(tensor.shape) == 4:
            imgs = []
            for i in range(tensor.shape[0]):
                imgs.extend(cls.tensor2img(tensor[i, ...]))
            return imgs
        assert tensor.shape[0] == 3
        img = np.transpose(tensor, (1, 2, 0))
        img = img * np.array(std) + np.array(mean)
        return [img*255]


    def __getitem__(self, i):
        # print('__getitem__\t', i, i%16, '\tlabel:', self.labels[i])
        #label = self.labels[i]
        img = Image.open(os.path.join(self.data_root, self.fns[i])).convert('RGB')
        img = self.transforms(img)
        return img if not self.return_fn else (img, self.fns[i])

class SourceSampler(torch.utils.data.Sampler):
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
        for i in range(self.__len__()):
            # sample both positive and negative labels
            pos_labels = np.random.choice(self.labels, int(self.batch_size/(2*self.batch_k)), replace=False)
            neg_labels = np.random.choice(self.labels, int(self.batch_size/(2*self.batch_k)), replace=False)
            ret_idx = []
            for label in pos_labels:
                ret_idx.extend(np.random.choice(self.data_source.idx[label], 2, replace=False))
                # print('\t\tpositive label ', label, '\t', ret_idx[-2:]) 
            for label in neg_labels:
                neg_label = np.random.choice([l for l in self.labels if l != label], 1)[0]
                label_idx = np.random.choice(self.data_source.idx[label], 1)
                neg_label_idx = np.random.choice(self.data_source.idx[neg_label], 1)
                ret_idx.extend([label_idx[0], neg_label_idx[0]])
            yield ret_idx


def main_train():
    from sampler import BalancedBatchSampler
    data = ImageFolderWithName(return_fn=False, root='CUB_100_train/images', transform=transformations)
    dataset = torch.utils.data.DataLoader(data, batch_sampler=BalancedBatchSampler(data, batch_size=32, batch_k=4, length=2000), num_workers=4, pin_memory=True)

    return dataset

def main_test():
    data = ImageFolderWithName(return_fn=False, root='CUB_100_test/images',transform=transformations)
    dataset = torch.utils.data.DataLoader(data)

    return dataset

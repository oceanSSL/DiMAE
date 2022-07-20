# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

import cv2
from PIL import Image
from PIL import ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import os


logger = getLogger()

memorycache = False
try:
    import mc, io
    memorycache = True
    print("using memory cache")
except:
    print("missing memory cache")
    pass

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


class MultiCropDatasetFile(torch.utils.data.Dataset):
    def __init__(
        self,
        image_file,
        meta_file,
        prefix,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        color_jitter_scale=1.0,
        return_index=False,
        pil_blur=True
    ):
        super(MultiCropDatasetFile, self).__init__()
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.image_file = image_file
        self.meta_file = meta_file
        self.prefix = prefix
        self.images, self.metas = self._read_dataset()
        self.samples = [(x, y) for (x, y) in zip(self.images, self.metas)]
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        trans = []
        color_transform = [get_color_distortion(s=color_jitter_scale), RandomGaussianBlur()]
        if pil_blur:
            color_transform = [get_color_distortion(s=color_jitter_scale), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

        self.initialized = False

    def _read_dataset(self):
        with open(self.image_file, 'r') as f:
            list_lines = f.readlines()
            list_lines = [x.strip() for x in list_lines]
            list_lines = [os.path.join(self.prefix, x) for x in list_lines]
            f.close()

        with open(self.meta_file, 'r') as f:
            meta_lines = f.readlines()
            meta_lines = [int(x.strip()) for x in meta_lines[1:]]
            meta_lines = np.array(meta_lines).astype(np.int)
            f.close()

        return list_lines, meta_lines

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        path, _ = self.samples[index]

        global memorycache
        if (not memorycache):
            image = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            image = pil_loader(value_str)

        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops

    def __len__(self):
        return len(self.samples)

class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        color_jitter_scale=1.0,
        return_index=False,
        pil_blur=True,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        trans = []
        color_transform = [get_color_distortion(s=color_jitter_scale), RandomGaussianBlur()]
        if pil_blur:
            color_transform = [get_color_distortion(s=color_jitter_scale), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        path, _ = self.samples[index]

        global memorycache
        if (not memorycache):
            image = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            image = pil_loader(value_str)

        # image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops

class MultiCropDatasetMetaWithDLabel(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file,
        prefix,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        color_jitter_scale=1.0,
        return_index=False,
        pil_blur=True
    ):
        super(MultiCropDatasetMeta, self).__init__()
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        # self.image_file = image_file
        self.meta_file = meta_file
        self.prefix = prefix
        self.images, self.metas, self.d_labels = self._read_dataset()
        self.samples = [(x, y, z) for (x, y, z) in zip(self.images, self.metas, self.d_labels)]
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        trans = []
        color_transform = [get_color_distortion(s=color_jitter_scale), RandomGaussianBlur()]
        if pil_blur:
            color_transform = [get_color_distortion(s=color_jitter_scale), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

        self.initialized = False

    def _read_dataset(self):
        with open(self.meta_file, 'r') as f:
            meta_lines = f.readlines()
            meta_lines = [x.strip() for x in meta_lines]
            image_path = [x.split(' ')[0] for x in meta_lines]
            lables = [int(x.split(' ')[1]) for x in meta_lines]
            d_labels = [int(x.split(' ')[2]) for x in meta_lines]
            f.close()

        return image_path, lables, d_labels

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        path, _, _ = self.samples[index]

        global memorycache
        if (not memorycache):
            image = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            image = pil_loader(value_str)

        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops

    def __len__(self):
        return len(self.samples)


class MultiCropDatasetMeta(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file,
        prefix,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        color_jitter_scale=1.0,
        return_index=False,
        pil_blur=True
    ):
        super(MultiCropDatasetMeta, self).__init__()
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        # self.image_file = image_file
        self.meta_file = meta_file
        self.prefix = prefix
        self.images, self.metas = self._read_dataset()
        self.samples = [(x, y) for (x, y) in zip(self.images, self.metas)]
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        trans = []
        color_transform = [get_color_distortion(s=color_jitter_scale), RandomGaussianBlur()]
        if pil_blur:
            color_transform = [get_color_distortion(s=color_jitter_scale), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

        self.initialized = False

    def _read_dataset(self):
        with open(self.meta_file, 'r') as f:
            meta_lines = f.readlines()
            meta_lines = [x.strip() for x in meta_lines]
            image_path = [x.split(' ')[0] for x in meta_lines]
            lables = [int(x.split(' ')[1]) for x in meta_lines]
            f.close()

        return image_path, lables

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        path, _ = self.samples[index]

        global memorycache
        if (not memorycache):
            image = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            image = pil_loader(value_str)

        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops

    def __len__(self):
        return len(self.samples)


class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class ImageFolderInstance(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(ImageFolderInstance, self).__init__(*args, **kwargs)
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        path, _ = self.samples[index]

        global memorycache
        if (not memorycache):
            sample = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            sample = pil_loader(value_str)

        # sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return index, sample


class MemoryCacheImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(MemoryCacheImageFolder, self).__init__(*args, **kwargs)
        self.initialized = False
        self.metas = [x[1] for x in self.samples]
        self.num_classes = np.max(self.metas) + 1
        assert self.num_classes == len(np.unique(self.metas)), 'Some labels skipped, please reorganize the labels!'

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):

        path, target = self.samples[index]

        global memorycache
        if (not memorycache):
            sample = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            sample = pil_loader(value_str)

        # sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class MemoryCacheImageFile(torch.utils.data.Dataset):
    def __init__(
        self,
        image_file,
        meta_file,
        prefix,
        transform=None,
        target_transform=None,
        return_index=False
    ):
        super(MemoryCacheImageFile, self).__init__()
        self.image_file = image_file
        self.meta_file = meta_file
        self.prefix = prefix
        self.images, self.metas = self._read_dataset()
        self.samples = [(x, y) for (x, y) in zip(self.images, self.metas)]
        self.num_classes = np.max(self.metas) + 1
        assert self.num_classes == len(np.unique(self.metas)), 'Some labels skipped, please reorganize the labels!'
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = return_index
        self.initialized = False

    def set_metas(self, meta_file):
        self.meta_file = meta_file
        with open(self.meta_file, 'r') as f:
            meta_lines = f.readlines()
            meta_lines = [int(x.strip()) for x in meta_lines[1:]]
            meta_lines = np.array(meta_lines).astype(np.int)
            f.close()
        self.samples = [(x, y) for (x, y) in zip(self.images, self.metas)]
        self.num_classes = np.max(self.metas) + 1
        assert self.num_classes == len(np.unique(self.metas)), 'Some labels skipped, please reorganize the labels!'
        # self.initialized = False

    def _read_dataset(self):
        with open(self.image_file, 'r') as f:
            list_lines = f.readlines()
            list_lines = [x.strip() for x in list_lines]
            list_lines = [os.path.join(self.prefix, x) for x in list_lines]
            f.close()

        with open(self.meta_file, 'r') as f:
            meta_lines = f.readlines()
            meta_lines = [int(x.strip()) for x in meta_lines[1:]]
            meta_lines = np.array(meta_lines).astype(np.int)
            f.close()

        return list_lines, meta_lines

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        path, target = self.samples[index]

        global memorycache
        if (not memorycache):
            sample = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            sample = pil_loader(value_str)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return sample, target, index
        else:
            return sample, target

class MemoryCacheImageMeta(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file,
        prefix,
        transform=None,
        target_transform=None,
        return_index=False
    ):
        super(MemoryCacheImageMeta, self).__init__()
        # self.image_file = image_file
        self.meta_file = meta_file
        self.prefix = prefix
        self.images, self.metas = self._read_dataset()
        self.samples = [(x, y) for (x, y) in zip(self.images, self.metas)]
        self.num_classes = np.max(self.metas) + 1
        assert self.num_classes == len(np.unique(self.metas)), 'Some labels skipped, please reorganize the labels!'
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = return_index
        self.initialized = False

    def _read_dataset(self):
        with open(self.meta_file, 'r') as f:
            meta_lines = f.readlines()
            meta_lines = [x.strip() for x in meta_lines]
            image_path = [x.split(' ')[0] for x in meta_lines]
            lables = [int(x.split(' ')[1]) for x in meta_lines]
            f.close()

        return image_path, lables

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        path, target = self.samples[index]
        path = self.prefix + path

        global memorycache
        if (not memorycache):
            sample = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            sample = pil_loader(value_str)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return sample, target, index
        else:
            return sample, target

class MemoryCacheImageMetaWithDlabel(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file,
        prefix,
        transform=None,
        target_transform=None,
        return_index=False
    ):
        super(MemoryCacheImageMetaWithDlabel, self).__init__()
        # self.image_file = image_file
        self.meta_file = meta_file
        self.prefix = prefix
        self.images, self.metas, self.d_labels = self._read_dataset()
        self.samples = [(x, y, z) for (x, y, z) in zip(self.images, self.metas, self.d_labels)]
        self.num_classes = np.max(self.metas) + 1
        assert self.num_classes == len(np.unique(self.metas)), 'Some labels skipped, please reorganize the labels!'
        self.transform = transform
        self.target_transform = target_transform
        self.initialized = False
        self.return_index = return_index

    def _read_dataset(self):
        with open(self.meta_file, 'r') as f:
            meta_lines = f.readlines()
            meta_lines = [x.strip() for x in meta_lines]
            image_path = [x.split(' ')[0] for x in meta_lines]
            lables = [int(x.split(' ')[1]) for x in meta_lines]
            d_labels = [int(x.split(' ')[2]) for x in meta_lines]
            f.close()

        return image_path, lables, d_labels

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        path, target, d_label = self.samples[index]

        global memorycache
        if (not memorycache):
            sample = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            sample = pil_loader(value_str)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return sample, target, d_label, index
        else:
            return sample, target, d_label

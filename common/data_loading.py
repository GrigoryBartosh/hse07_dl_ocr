import os

import numpy as np

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO

import common.simple_stacker as simple_stacker
from common.config import PATH

__all__ = ['get_loader']


class ImageGetter():
    def __init__(self, transform):
        self.transform = transform

    def get(self, path):
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image


class DatasetImages(data.Dataset):
    def _check_filename(filename):
        return filename.endswith('.png') or filename.endswith('.jpg')

    def __init__(self, imgs_dir, transform=None):
        self.paths = []
        for (dirpath, dirnames, filenames) in os.walk(imgs_dir):
            filenames = filter(DatasetImages._check_filename, filenames)
            filenames = map(lambda f: os.path.join(dirpath, f), filenames)
            self.paths += list(filenames)

        self.image_getter = ImageGetter(transform)

    def __getitem__(self, index):
        path = self.paths[index]
        return self.image_getter.get(path)

    def __len__(self):
        return len(self.paths)


class DatasetTextCOCO(data.Dataset):
    def __init__(self, captions_path):
        self.coco = COCO(captions_path)
        self.ids = list(self.coco.anns.keys())

    def __getitem__(self, index):
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']

        return caption

    def __len__(self):
        return len(self.ids)


class DatasetTextSampler(data.Dataset):
    def __init__(self, dataset, image_size):
        self.dataset = dataset
        self.image_size = image_size

    def _sample_params(self):
        return np.random.uniform(-1, 1, 8).tolist()

    def __getitem__(self, index):
        text = self.dataset[index]
        params = self._sample_params()
        image, _ = simple_stacker.stack(text, self.image_size)
        image_t, bounding_boxes = simple_stacker.stack(text, self.image_size, *params[:5])

        image = torch.tensor(image, dtype=torch.float32)
        image_t = torch.tensor(image_t, dtype=torch.float32)
        params = torch.tensor(params, dtype=torch.float32)
        bounding_boxes = bounding_boxes # TODO

        return text, image, image_t, params, bounding_boxes

    def __len__(self):
        return len(self.dataset)


def get_image_transform(image_size):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], 
        std=[0.5, 0.5, 0.5]
    )

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def collate_images(images):
    images = torch.stack(images, 0)
    return images


def collate_params(params):
    params = torch.stack(params, 0)
    return params


def collate_bounding_boxes(bounding_boxes): # TODO
    return bounding_boxes


def collate_texts(data):
    texts, x_t, x_tp, x_p, x_bb = zip(*data)

    x_t = collate_images(x_t)
    x_tp = collate_images(x_tp)
    x_p = collate_params(x_p)
    x_bb = collate_bounding_boxes(x_bb)

    return texts, x_t, x_tp, x_p, x_bb


def infinit_data_loader(data_loader):
    while True:
        for x in data_loader:
            yield x


def union_data_loaders(data_loader_a, data_loader_b):
    for (a, b) in zip(data_loader_a, data_loader_b):
        yield a, b


def get_loader(sset='VAL', image_transform=None, image_size=256, 
               batch_size=16, shuffle=False, num_workers=8):
    if image_transform is None:
        image_transform = get_image_transform(image_size)

    dataset_images = DatasetImages(
        imgs_dir=PATH['DATASETS']['COCO'][sset]['IMAGES_DIR'],
        transform=image_transform
    )
    dataset_texts = DatasetTextSampler(
        DatasetTextCOCO(captions_path=PATH['DATASETS']['COCO'][sset]['CAPTIONS']),
        image_size
    )

    data_loader_images = data.DataLoader(
        dataset=dataset_images,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(1, num_workers // 2),
        collate_fn=collate_images,
        drop_last=True
    )
    data_loader_texts = data.DataLoader(
        dataset=dataset_texts,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(1, num_workers // 2),
        collate_fn=collate_texts,
        drop_last=True
    )

    data_loader_images = infinit_data_loader(data_loader_images)
    data_loader_texts = infinit_data_loader(data_loader_texts)
    data_loader = union_data_loaders(data_loader_images, data_loader_texts)

    return data_loader

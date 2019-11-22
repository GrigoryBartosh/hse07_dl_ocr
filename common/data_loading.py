import os

import numpy as np

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image

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


# TODO @iisuslik43
class TextEncoder():
    def __init__(self):
        pass

    def encode(self, text):
        return text


class DatasetImages(data.Dataset):
    def _check_filename(filename):
        return filename[-4:] == '.png' or filename[-4:] == '.jpg'

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


# TODO @grigorybartosh
class DatasetText(data.Dataset):
    def __init__(self, captions_path):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


# TODO @grigorybartosh
class DatasetVK(data.Dataset):
    def __init__(self, imgs_dir, captions_path, image_transform=None):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def get_image_transform():
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ])


def collate_images(images):
    images = torch.stack(images, 0)
    return images


def collate_texts(texts):
    # TODO @iisuslik43 @grigorybartosh
    return texts


def collate_parallel(data):
    images, captions = zip(*data)

    images = collate_images(images)
    captions = collate_texts(captions)

    return images, captions


def infinit_data_loader(data_loader):
    while True:
        for x in data_loader:
            yield x


def union_data_loaders(data_loader_a, data_loader_b):
    for (a, b) in zip(data_loader_a, data_loader_b):
        yield a, b


def get_data_loader_images(dataset_name, sset, transform, batch_size, shuffle, num_workers):
    assert dataset_name in PATH['DATASETS'], f"Unknown dataset_name value '{dataset_name}'"

    dataset_images = DatasetImages(
        imgs_dir=PATH['DATASETS'][dataset_name][sset]['IMAGES_DIR'],
        transform=transform
    )

    data_loader_images = data.DataLoader(
        dataset=dataset_images,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_images,
        drop_last=True
    )

    return infinit_data_loader(data_loader_images)


# TODO @grigorybartosh
def get_data_loader_texts(dataset_name, sset, batch_size, shuffle, num_workers):
    pass


# TODO @grigorybartosh
def get_data_loader_parallel(
    dataset_name, sset, image_transform, batch_size, shuffle, num_workers):
    pass


def get_loader(dataset_images_name=None, dataset_texts_name=None,
               dataset_parallel_name=None,
               sset='VAL', image_transform=None, batch_size=16, shuffle=False, num_workers=8):
    if image_transform is None:
        image_transform = get_image_transform()

    if dataset_images_name or dataset_texts_name:
        data_loader_images = get_data_loader_images(
            dataset_images_name,
            sset,
            image_transform,
            batch_size,
            shuffle,
            max(1, num_workers // 2) if dataset_texts_name else num_workers
        )

        data_loader_texts = get_data_loader_texts(
            dataset_texts_name,
            sset,
            batch_size,
            shuffle,
            max(1, num_workers // 2) if dataset_images_name else num_workers
        )

        data_loader = union_data_loaders(data_loader_images, data_loader_texts)

    if dataset_parallel_name:
        data_loader_parallel = get_data_loader_parallel(
            dataset_parallel_name,
            sset,
            image_transform,
            batch_size,
            shuffle,
            num_workers
        )

        data_loader = data_loader_parallel

    return data_loader

import os

import numpy as np

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO

import common.simple_stacker as simple_stacker
import common.utils as utils
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
    def __init__(self, dataset, image_size, params_count, params_move_count):
        self.dataset = dataset
        self.image_size = image_size
        self.params_count = params_count
        self.params_move_count = params_move_count

        self.box_encoder = utils.BoxEncoder()

    def _sample_params(self):
        return np.random.uniform(-1, 1, self.params_count).tolist()

    def _sample_text(self, text):
        text = ''.join([c for c in text if utils.is_char_valid(c) or c == ' '])
        text = text.split()
        n = np.random.randint(1, min(6, len(text) + 1))
        l = np.random.randint(0, len(text) - n + 1)
        text = ' '.join(text[l:l + n])
        return text

    def __getitem__(self, index):
        text = self._sample_text(self.dataset[index])

        params = self._sample_params()

        image, _, _ = simple_stacker.stack(text, self.image_size)

        image_t, bboxes, labels = simple_stacker.stack(
            text,
            self.image_size, 
            *params[:self.params_move_count]
        )
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.LongTensor(labels)
        bboxes, labels = self.box_encoder.encode(bboxes, labels)

        image = torch.tensor(image, dtype=torch.float32)
        image_t = torch.tensor(image_t, dtype=torch.float32)
        params = torch.tensor(params, dtype=torch.float32)

        return text, image, image_t, params, bboxes, labels

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


def collate_bounding_boxes(bboxes):
    bboxes = torch.stack(bboxes, 0)
    return bboxes


def collate_texts(data):
    texts, x_t, x_tp, x_p, x_bb, x_l = zip(*data)

    x_t = torch.stack(x_t)
    x_tp = torch.stack(x_tp)
    x_p = torch.stack(x_p)
    x_bb = torch.stack(x_bb)
    x_l = torch.stack(x_l)

    return texts, x_t, x_tp, x_p, x_bb, x_l


def infinit_data_loader(data_loader):
    while True:
        for x in data_loader:
            yield x


def union_data_loaders(data_loader_a, data_loader_b):
    for (a, b) in zip(data_loader_a, data_loader_b):
        yield a, b


def get_loader(sset='VAL', image_transform=None, image_size=256,
               params_count=8, params_move_count=5,
               batch_size=16, shuffle=False, num_workers=8):
    if image_transform is None:
        image_transform = get_image_transform(image_size)

    dataset_images = DatasetImages(
        imgs_dir=PATH['DATASETS']['COCO'][sset]['IMAGES_DIR'],
        transform=image_transform
    )
    dataset_texts = DatasetTextSampler(
        DatasetTextCOCO(captions_path=PATH['DATASETS']['COCO'][sset]['CAPTIONS']),
        image_size,
        params_count,
        params_move_count
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

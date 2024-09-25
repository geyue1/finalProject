# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> prepare_AAR_ag.py
@Author : yge
@Date : 2024/6/30 16:08
@Desc :

==============================================================
'''
import os
import random

import numpy as np
from torch.utils.data import ConcatDataset
from torchvision import transforms, datasets

from src.util.utils_ import local_dataset

def _save_images_to_folder(dataset, transform, path, format_='.png'):
    scales = {}
    for id,el in enumerate(dataset):
        img = transform(el[0])
        out = os.path.join(path, str(el[1]))
        if not os.path.exists(out):
            os.makedirs(out)
        img_path = os.path.join(out, str(id)+format_)
        img.save(img_path)
def make_aar_augment( min_scale=0.5, max_scale=1.5, seed=0, **kwargs):
    data_path = os.path.join("..","..", "data", "AAR")

    np.random.seed(seed)
    transform = transforms.Compose([
        transforms.RandomAffine(0, scale=(min_scale, max_scale)),
        transforms.RandomRotation(degrees=(-15,15))
    ])
    #transform = transforms.RandomAffine(0, scale=(min_scale, max_scale))
    #transforms.RandomRotation

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    #samples = random.Random.sample(population=concat_dataset,k=RANDOM_SIZE)
    dataset_path = os.path.join("..","..","data", 'AAR_augment')

    idx = _save_images_to_folder(dataset, transform, dataset_path,  '.png')

if __name__ == '__main__':
    make_aar_augment()
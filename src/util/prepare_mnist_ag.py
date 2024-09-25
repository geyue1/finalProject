# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> prepare_mnist_ag.py
@Author : yge
@Date : 2024/7/19 21:55
@Desc :

==============================================================
'''
import os
import random

import numpy as np
from torch.utils.data import ConcatDataset
from torchvision import transforms, datasets


def _save_images_to_folder(dataset, transform, path, format_='.png'):
    scales = {}
    for id,el in enumerate(dataset):
        img = transform(el[0])

        out = os.path.join(path, str(el[1]))
        if not os.path.exists(out):
            os.makedirs(out)
        img_path = os.path.join(out, str(id)+format_)
        img.save(img_path)

def make_mnist_augment(source, dest, min_scale=0.5, max_scale=1.5, download=True, seed=0, **kwargs):



    RANDOM_SIZE = 1

    np.random.seed(seed)


    transform = transforms.Compose([
        transforms.RandomAffine(0, scale=(min_scale, max_scale)),
        transforms.RandomRotation(degrees=(-15,15))
    ])
    transform0 = transforms.Compose([

    ])

    #transform = transforms.RandomAffine(0, scale=(min_scale, max_scale))
    #transforms.RandomRotation

    dataset_train = datasets.MNIST(root=source, train=True, download=download)
    dataset_test = datasets.MNIST(root=source, train=False, download=download)
    concat_dataset = ConcatDataset([dataset_train, dataset_test])

    samples = random.sample([ el for el in concat_dataset],RANDOM_SIZE)
    #samples = random.Random.sample(population=concat_dataset,k=RANDOM_SIZE)
    dataset_path = os.path.join("..","..","data1", 'MNIST_augment')
    dataset_path0 = os.path.join("..", "..", "data1", 'MNIST')

    idx = _save_images_to_folder(samples, transform, dataset_path,  '.png')
    _save_images_to_folder(samples, transform0, dataset_path0, '.png')


if __name__ == '__main__':
    '''
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='source folder of the dataset')
    parser.add_argument('--dest', type=str, required=True, help='destination folder for the output')
    parser.add_argument('--min_scale', type=float, required=True,
                        help='min scale for the generated dataset')
    parser.add_argument('--max_scale', type=float, default=1.5,
                        help='max scale for the generated dataset')
    # parser.add_argument('--download', action='store_true',
    #                     help='donwload stource dataset if needed.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('{}={}'.format(k, v))

    make_mnist_augment(**vars(args))

'''
    make_mnist_augment(source="../../data",dest="../../data")
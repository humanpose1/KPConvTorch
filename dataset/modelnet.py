import numpy as np
from loguru import logger
import os
import os.path as osp
import pickle
import time
import torch
from torch.utils.data import Dataset
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.pool import pool_pos
from tqdm import tqdm
import objgraph
from pympler.tracker import SummaryTracker
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

from dataset.data import MultiScaleData


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)



def torch_subsampling(data, subsampling_parameter):
    batch = torch.zeros(data.shape[0], dtype=torch.long)
    pool = voxel_grid(torch.tensor(data[:, :3]), batch, subsampling_parameter)
    points = pool_pos(pool, torch.tensor(data[:, :3]))
    normals = pool_pos(pool, torch.tensor(data[:, 3:]))
    del pool
    del batch
    return points.numpy(), normals.numpy()


def load_subsampled_clouds(subsampling_parameter, path,
                           data_folder):
    """
    source : https://github.com/HuguesTHOMAS/KPConv/blob/master/datasets/ModelNet40.py
    """
    label_to_names = {0: 'airplane',
                      1: 'bathtub',
                      2: 'bed',
                      3: 'bench',
                      4: 'bookshelf',
                      5: 'bottle',
                      6: 'bowl',
                      7: 'car',
                      8: 'chair',
                      9: 'cone',
                      10: 'cup',
                      11: 'curtain',
                      12: 'desk',
                      13: 'door',
                      14: 'dresser',
                      15: 'flower_pot',
                      16: 'glass_box',
                      17: 'guitar',
                      18: 'keyboard',
                      19: 'lamp',
                      20: 'laptop',
                      21: 'mantel',
                      22: 'monitor',
                      23: 'night_stand',
                      24: 'person',
                      25: 'piano',
                      26: 'plant',
                      27: 'radio',
                      28: 'range_hood',
                      29: 'sink',
                      30: 'sofa',
                      31: 'stairs',
                      32: 'stool',
                      33: 'table',
                      34: 'tent',
                      35: 'toilet',
                      36: 'tv_stand',
                      37: 'vase',
                      38: 'wardrobe',
                      39: 'xbox'}
    name_to_label = {v: k for k, v in label_to_names.items()}

    if 0 < subsampling_parameter <= 0.01:
        raise ValueError('subsampling_parameter too low (should be over 1 cm')

    # Initiate containers
    input_points = {'training': [], 'validation': [], 'test': []}
    input_normals = {'training': [], 'validation': [], 'test': []}
    input_labels = {'training': [], 'validation': []}

    ################
    # Training files
    ################

    # Restart timer
    t0 = time.time()

    # Load wanted points if possible
    logger.info('\nLoading training points')
    filename = osp.join(path,
                        'train_{:.3f}_record.pkl'.format(
                            subsampling_parameter))

    if osp.exists(filename):
        with open(filename, 'rb') as file:
            input_points['training'], \
                input_normals['training'], \
                input_labels['training'] = pickle.load(file)

    # Else compute them from original points
    else:

        # Collect training file names
        names = np.loadtxt(osp.join(path, data_folder, 'modelnet40_train.txt'),
                           dtype=np.str)

        # Collect point clouds
        for i, cloud_name in tqdm(enumerate(names), total=len(names)):

            # Read points
            class_folder = '_'.join(cloud_name.split('_')[:-1])
            txt_file = osp.join(path,
                                data_folder,
                                class_folder,
                                cloud_name) + '.txt'
            data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

            # Subsample them
            if subsampling_parameter > 0:
                # ints, normals = torch_subsampling(data,subsampling_parameter)
                points, normals = grid_subsampling(data[:, :3],
                                                   features=data[:, 3:],
                                                   sampleDl=subsampling_parameter)
            else:
                points = data[:, :3]
                normals = data[:, 3:]

            # Add to list
            input_points['training'] += [points]
            input_normals['training'] += [normals]

        # Get labels
        label_names = ['_'.join(name.split('_')[:-1]) for name in names]
        input_labels['training'] = np.array(
            [name_to_label[name] for name in label_names])

        # Save for later use
        with open(filename, 'wb') as file:
            pickle.dump((input_points['training'],
                         input_normals['training'],
                         input_labels['training']), file)

    lengths = [p.shape[0] for p in input_points['training']]
    sizes = [l * 4 * 6 for l in lengths]
    logger.info('{:.1f} MB loaded in {:.1f}s'.format(
        np.sum(sizes) * 1e-6, time.time() - t0))

    ############
    # Test files
    ############

    # Restart timer
    t0 = time.time()

    # Load wanted points if possible
    logger.info('\nLoading test points')
    filename = osp.join(path, 'test_{:.3f}_record.pkl'.format(
        subsampling_parameter))
    if osp.exists(filename):
        with open(filename, 'rb') as file:
            input_points['validation'], \
                input_normals['validation'], \
                input_labels['validation'] = pickle.load(file)

    # Else compute them from original points
    else:

        # Collect test file names
        names = np.loadtxt(osp.join(path, data_folder,
                                    'modelnet40_test.txt'), dtype=np.str)

        # Collect point clouds
        for i, cloud_name in tqdm(enumerate(names), total=len(names)):

            # Read points
            class_folder = '_'.join(cloud_name.split('_')[:-1])
            txt_file = osp.join(path, data_folder,
                                class_folder, cloud_name) + '.txt'
            data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

            # Subsample them
            if subsampling_parameter > 0:
                points, normals = torch_subsampling(data,
                                                    subsampling_parameter)
            else:
                points = data[:, :3]
                normals = data[:, 3:]

            # Add to list
            input_points['validation'] += [points]
            input_normals['validation'] += [normals]

        # Get labels
        label_names = ['_'.join(name.split('_')[:-1]) for name in names]
        input_labels['validation'] = np.array(
            [name_to_label[name] for name in label_names])

        # Save for later use
        # Save for later use
        with open(filename, 'wb') as file:
            pickle.dump((input_points['validation'],
                         input_normals['validation'],
                         input_labels['validation']), file)

    lengths = [p.shape[0] for p in input_points['validation']]
    sizes = [l * 4 * 6 for l in lengths]
    logger.info('{:.1f} MB loaded in {:.1f}s\n'.format(
        np.sum(sizes) * 1e-6, time.time() - t0))

    # Test = validation
    input_points['test'] = input_points['validation']
    input_normals['test'] = input_normals['validation']
    input_labels['test'] = input_labels['validation']

    return input_points, input_normals, input_labels


class ModelNet(Dataset):
    """
    modelnet40 dataset
    """

    def __init__(self, path, subsampling_parameter, split='training',
                 transforms=None, num_features=1):

        self.path = path
        self.data_folder = 'modelnet40_normal_resampled'
        self.num_train = 9843
        self.num_test = 2468
        self.num_features = num_features

        # load_subsampled_clouds(subsampling_parameter)
        self.split = split
        self.transforms = transforms
        self.input_points, self.input_normals, self.input_labels = load_subsampled_clouds(subsampling_parameter, path, self.data_folder)

    def __len__(self):
        if(self.split == 'training'):
            return self.num_train
        else:
            return self.num_test

    def __getitem__(self, idx):

        points = self.input_points[self.split][idx].astype(np.float32)
        normals = self.input_normals[self.split][idx].astype(np.float32)
        labels = int(self.input_labels[self.split][idx])

        if(self.num_features == 1):
            x = torch.ones(points.shape[0], 1)
        elif self.num_features == 3:
            x = torch.from_numpy(normals)

        data = MultiScaleData(pos=torch.from_numpy(points),
                              x=x,
                              y=labels)
        if(self.transforms is not None):
            data = self.transforms(data)
        return data

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

from dataset.data import MultiScaleData


def torch_subsampling(data, subsampling_parameter):
    batch = torch.zeros(data.shape[0], dtype=torch.long)
    pool = voxel_grid(torch.tensor(data[:, :3]), batch, subsampling_parameter)
    points = pool_pos(pool, torch.tensor(data[:, :3]).numpy())
    normals = pool_pos(pool, torch.tensor(data[:, 3:]).numpy())
    return points, normals


class ModelNet(Dataset):
    """
    modelnet40 dataset
    """

    def __init__(self, path, subsampling_parameter, split='train',
                 transforms=None, num_features=1):

        self.path = path
        self.data_folder = 'modelnet40_normal_resampled'
        self.num_train = 9843
        self.num_test = 2468
        self.num_features = num_features

        self.load_subsampled_clouds(subsampling_parameter)
        self.split = split
        self.transforms = transforms

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        source : https://github.com/HuguesTHOMAS/KPConv/blob/master/datasets/ModelNet40.py
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_points = {'training': [], 'validation': [], 'test': []}
        self.input_normals = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        ################
        # Training files
        ################

        # Restart timer
        t0 = time.time()
        logger.info("Loading training points")
        filename = osp.join(
            self.path,
            'train_{:.3f}_record.pkl'.format(self.subsampling_parameter))
        if(osp.exists(filename)):
            with open(filename, 'rb') as f:
                self.input_points['training'], \
                    self.input_normals['training'], \
                    self.input_labels['training'] = pickle.load(f)

        else:
            # We need to create the pickle file
            names = np.loadtxt(osp.join(self.path,
                                        self.data_folder,
                                        'modelnet40_train.txt'),
                               dtype=np.str)

            for i, cloud_name in enumerate(names):
                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = osp.join(self.path,
                                    self.data_folder,
                                    class_folder,
                                    cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)
                if(self.subsampling_parameter > 0):

                    points, normals = torch_subsampling(data,
                                                        subsampling_parameter)
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                # Add to list
                self.input_points['training'] += [points]
                self.input_normals['training'] += [normals]
            # Get labels
            label_names = ['_'.join(name.split('_')[:-1]) for name in names]
            self.input_labels['training'] = np.array([self.name_to_label[name] for name in label_names])

            # Save for later use
            with open(filename, 'wb') as f:
                pickle.dump((self.input_points['training'],
                             self.input_normals['training'],
                             self.input_labels['training']), f)
        lengths = [p.shape[0] for p in self.input_points['training']]
        sizes = [l * 4 * 6 for l in lengths]
        logger.info('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))


        ############
        # Test files
        ############

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        logger.info('\nLoading test points')
        filename = osp.join(self.path, 'test_{:.3f}_record.pkl'.format(
            self.subsampling_parameter))

        if osp.exists(filename):
            with open(filename, 'rb') as f:
                self.input_points['validation'], \
                self.input_normals['validation'], \
                self.input_labels['validation'] = pickle.load(f)

        # Else compute them from original points
        else:

            # Collect test file names
            names = np.loadtxt(osp.join(
                self.path, self.data_folder,
                'modelnet40_test.txt'), dtype=np.str)

            # Collect point clouds
            for i, cloud_name in enumerate(names):

                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = osp.join(
                    self.path, self.data_folder, class_folder,
                    cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

                # Subsample them
                if subsampling_parameter > 0:
                    points, normals = torch_subsampling(data,
                                                        subsampling_parameter)
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                # Add to list
                self.input_points['validation'] += [points]
                self.input_normals['validation'] += [normals]


            # Get labels
            label_names = ['_'.join(name.split('_')[:-1]) for name in names]
            self.input_labels['validation'] = np.array([self.name_to_label[name] for name in label_names])

            # Save for later use
            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_points['validation'],
                             self.input_normals['validation'],
                             self.input_labels['validation']), file)

        lengths = [p.shape[0] for p in self.input_points['validation']]
        sizes = [l * 4 * 6 for l in lengths]
        logger.info('{:.1f} MB loaded in {:.1f}s\n'.format(
            np.sum(sizes) * 1e-6, time.time() - t0))

        small = False
        if small:

            for split in ['training', 'validation']:

                pick_n = 50
                gen_indices = []
                for l in self.label_values:
                    label_inds = np.where(np.equal(self.input_labels[split], l))[0]
                    if len(label_inds) > pick_n:
                        label_inds = label_inds[:pick_n]
                    gen_indices += [label_inds.astype(np.int32)]
                gen_indices = np.hstack(gen_indices)

                self.input_points[split] = np.array(
                    self.input_points[split])[gen_indices]
                self.input_normals[split] = np.array(
                    self.input_normals[split])[gen_indices]
                self.input_labels[split] = np.array(
                    self.input_labels[split])[gen_indices]

                if split == 'training':
                    self.num_train = len(gen_indices)
                else:
                    self.num_test = len(gen_indices)

        # Test = validation
        self.input_points['test'] = self.input_points['validation']
        self.input_normals['test'] = self.input_normals['validation']
        self.input_labels['test'] = self.input_labels['validation']

    def __len__(self):
        if(self.split == 'train'):
            return self.num_train
        else:
            return self.num_test

    def __getitem__(self, idx):

        points = self.input_points[self.split][idx].astype(np.float32)
        normals = self.input_normals[self.split][idx].astype(np.float32)
        if(self.num_features == 1):
            x = torch.ones(points.shape[0], 1)
        elif self.num_features == 3:
            x = torch.from_numpy(normals)

        data = MultiScaleData(pos=torch.from_numpy(points), x=x)
        if(self.transforms is not None):
            data = self.transforms(data)
        return data

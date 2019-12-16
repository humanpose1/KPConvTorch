# Kernel Point Convolution in Pytorch

import torch
from torch.nn import Parameter

from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_max


from kernels.kernel_points import load_kernels as create_kernel_points
from kernels.convolution_ops import KPConv_ops
from kernels.convolution_ops import KPConv_deform_ops
from models.utilities import weight_variable


class KPConvLayer(torch.nn.Module):
    """
    apply the kernel point convolution on a point cloud

    NB : it is the original version of KPConv, it is not the message passing version
    attributes:
    layer_ind (int): index of the layer
    radius: radius of the kernel
    num_inputs : dimension of the input feature
    num_outputs : dimension of the output feature
    config : YACS class that contains all the important constants
    and hyperparameters
    """

    def __init__(self, radius, num_inputs, num_outputs, config):
        super(KPConvLayer, self).__init__()
        self.radius = radius
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config
        self.extent = self.config.NETWORK.KP_EXTENT * self.radius /\
            self.config.NETWORK.DENSITY_PARAMETER

        # Initial kernel extent for this layer
        K_radius = 1.5 * self.extent
        K_points_numpy = create_kernel_points(
            K_radius,
            self.config.NETWORK.NUM_KERNEL_POINTS,
            num_kernels=1,
            dimension=self.config.INPUT.POINTS_DIM,
            fixed=self.config.NETWORK.FIXED_KERNEL_POINTS)

        self.K_points = Parameter(torch.from_numpy(K_points_numpy.reshape((
            self.config.NETWORK.NUM_KERNEL_POINTS,
            self.config.INPUT.POINTS_DIM))).to(torch.float),
                                  requires_grad=False)

        self.weight = Parameter(
            weight_variable([self.config.NETWORK.NUM_KERNEL_POINTS,
                             self.num_inputs,
                             self.num_outputs]))

    def forward(self, pos, neighbors, x):
        """
        - pos is a tuple containing:
          - query_points(torch Tensor): query of size N x 3
          - support_points(torch Tensor): support points of size N0 x 3
        - neighbors(torch Tensor): neighbors of size N0 x M
        - features : feature of size N x d (d is the number of inputs)
        """
        support_points, query_points = pos
        new_feat = KPConv_ops(query_points,
                              support_points,
                              neighbors,
                              x,
                              self.K_points,
                              self.weight,
                              self.extent,
                              self.config.NETWORK.KP_INFLUENCE,
                              self.config.NETWORK.CONVOLUTION_MODE)
        return new_feat


class DeformableKPConvLayer(torch.nn.Module):

    def __init__(self, radius, num_inputs,
                 num_outputs, config, version=0, modulated=False):
        """
        it doesn't work yet :
        """
        super(DeformableKPConvLayer, self).__init__()
        self.radius = radius
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config
        self.extent = self.config.NETWORK.KP_EXTENT * self.radius /\
            self.config.NETWORK.DENSITY_PARAMETER
        self.version = version
        self.modulated = modulated

        # Initial kernel extent for this layer
        K_radius = 1.5 * self.extent
        K_points_numpy = create_kernel_points(
            K_radius,
            self.config.NETWORK.NUM_KERNEL_POINTS,
            num_kernels=1,
            dimension=self.config.INPUT.POINTS_DIM,
            fixed=self.config.NETWORK.FIXED_KERNEL_POINTS)

        self.K_points = Parameter(torch.from_numpy(K_points_numpy.reshape((
            self.config.NETWORK.NUM_KERNEL_POINTS,
            self.config.INPUT.POINTS_DIM))).to(torch.float),
                                  requires_grad=False)

        # Parameter of the deformable convolution
        self.weight = Parameter(
            weight_variable([self.config.NETWORK.NUM_KERNEL_POINTS,
                             self.num_inputs,
                             self.num_outputs]))
        if(self.modulated):
            offset_dim = (self.config.INPUT.POINTS_DIM+1) * (self.config.NETWORK.NUM_KERNEL_POINTS -1)
        else:
            offset_dim = (self.config.INPUT.POINTS_DIM) * (self.config.NETWORK.NUM_KERNEL_POINTS -1)

        if(self.version == 0):
            # kp conv to estimate the offset
            self.deformable_weight = Parameter(
                weight_variable([self.config.NETWORK.NUM_KERNEL_POINTS,
                                 self.num_inputs,
                                 offset_dim]))
        elif(self.version == 1):
            # MLP to estimate the offset
            self.deformable_weight = Parameter(
                weight_variable([self.num_inputs,
                                 offset_dim]))
        self.bias = torch.nn.Parameter(
            torch.zeros(offset_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, query_points, support_points, neighbors, features):

        points_dim = self.config.INPUT.POINTS_DIM
        num_kpoints = self.config.NETWORK.NUM_KERNEL_POINTS
        if(self.version == 0):
            features0 = KPConv_ops(query_points,
                                   support_points,
                                   neighbors,
                                   features,
                                   self.K_points,
                                   self.deformable_weight,
                                   self.extent,
                                   self.config.NETWORK.KP_INFLUENCE,
                                   self.config.NETWORK.CONVOLUTION_MODE) + self.bias

        if self.modulated:
            # Get offset (in normalized scale) from features
            offsets = features0[:, :points_dim * (num_kpoints - 1)]
            offsets = offsets.reshape([-1, (num_kpoints - 1), points_dim])

            # Get modulations
            modulations = 2 * torch.sigmoid(
                features0[:, points_dim * (num_kpoints - 1):])

            #  No offset for the first Kernel points
            if(self.version == 1):
                offsets = torch.cat([torch.zeros_like(offsets[:, :1, :]),
                                     offsets], axis=1)
                modulations = torch.cat([torch.zeros_like(modulations[:, :1]),
                                         modulations], axis=1)
        else:
            # Get offset (in normalized scale) from features
            offsets = features0.reshape([-1, (num_kpoints - 1), points_dim])
            # No offset for the first Kernel points
            offsets = torch.cat(
                [torch.zeros_like(offsets[:, :1, :]), offsets],
                axis=1)

            # No modulations
            modulations = None

        # Rescale offset for this layer
        offsets *= self.config.NETWORK.KP_EXTENT
        feat, sq_distances, _ = KPConv_deform_ops(query_points,
                                                  support_points,
                                                  neighbors,
                                                  features,
                                                  self.K_points,
                                                  offsets,
                                                  modulations,
                                                  self.weight,
                                                  self.extent,
                                                  self.config.NETWORK.KP_INFLUENCE,
                                                  self.config.NETWORK.CONVOLUTION_MODE)
        self.sq_distances = torch.nn.Parameter(sq_distances)
        return feat


class UnaryConv(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs, config):
        """
        1x1 convolution on point cloud (we can even call it a mini pointnet)

        """
        super(UnaryConv, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.config = config
        self.weight = Parameter(weight_variable([self.num_inputs,
                                                 self.num_outputs]))

    def forward(self, features):
        """
        features(Torch Tensor): size N x d d is the size of inputs
        """
        return torch.matmul(features, self.weight)


def max_pool(features, pools):

    if(pools.shape[1] > 2):
        x = torch.cat([features, torch.min(features, axis=0).values.view(1, -1)], axis=0)
        pool_features = x[pools]
        return torch.max(pool_features, axis=1).values
    else:
        row, col = pools.t()
        pool_features, _ = scatter_max(features[col], row, dim=0)

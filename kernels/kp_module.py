# Version of KPConv using message passing.
# Taken from https://github.com/nicolas-chaulet/deeppointcloud-benchmarks
from kernels.kernel_points import kernel_point_optimization_debug
from torch_geometric.nn import MessagePassing
import numpy as np
import torch
from torch.nn import Parameter


class PointKernel(MessagePassing):

    '''
    Implements KPConv: Flexible and Deformable Convolution for Point Clouds from
    https://arxiv.org/abs/1904.08889
    '''

    def __init__(self, num_points, in_features, out_features, radius=1,
                 kernel_dim=3, fixed='center', ratio=1, KP_influence='linear'):
        super(PointKernel, self).__init__()
        # PointKernel parameters
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points
        self.radius = radius
        self.kernel_dim = kernel_dim
        self.fixed = fixed
        self.ratio = ratio
        self.KP_influence = KP_influence
        # Radius of the initial positions of the kernel points
        self.KP_extent = radius / 1.5

        # Point position in kernel_dim
        self.kernel = Parameter(torch.Tensor(1, num_points, kernel_dim))

        # Associated weights
        self.kernel_weight = Parameter(torch.Tensor(num_points, in_features,
                                                    out_features))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.kernel_weight, a=np.sqrt(5))

        # Init the kernel using attrative + repulsion forces
        kernel, _ = kernel_point_optimization_debug(
            self.radius, self.num_points, num_kernels=1,
            dimension=self.kernel_dim, fixed=self.fixed, ratio=self.ratio,
            verbose=False)

        self.kernel.data = torch.from_numpy(kernel)

    def forward(self, x, pos, neighbors):
        print(x.shape)
        return self.propagate(neighbors.t(), x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j):

        if x_j is None:
            x_j = pos_j

        # Center every neighborhood [SUM n_neighbors(n_points), dim]
        neighbors = (pos_j - pos_i)

        # Number of points
        n_points = neighbors.shape[0]

        # Get points kernels
        K_points = self.kernel

        # Get all difference matrices [SUM n_neighbors(n_points), n_kpoints, dim]
        neighbors = neighbors.unsqueeze(1)

        differences = neighbors - K_points.float().view((-1, 3)).unsqueeze(0)
        sq_distances = (differences**2).sum(-1)

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.relu(1. - (sq_distances / (self.KP_extent ** 2)))
            # all_weights[all_weights < 0] = 0.0
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        neighbors_1nn = torch.argmin(sq_distances, dim=-1)
        weights = all_weights.gather(1, neighbors_1nn.unsqueeze(-1))

        K_weights = self.kernel_weight

        K_weights = torch.index_select(K_weights, 0, neighbors_1nn.view(-1)
                                       ).view((n_points,
                                               self.in_features,
                                               self.out_features))

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        features = x_j

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.einsum("nb, nc -> nc", weights, features)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        out = torch.einsum("na, nac -> nc", weighted_features, K_weights)
        out = out.view(-1, self.out_features)
        # import pdb; pdb.set_trace()
        return out

    def update(self, aggr_out, pos):
        return aggr_out

    def __repr__(self):
        # PointKernel parameters
        return "PointKernel({}, {}, {}, {}, {})".format(
            self.in_features,
            self.out_features, self.num_points, self.radius, self.KP_influence)

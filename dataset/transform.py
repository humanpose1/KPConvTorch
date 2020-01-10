import torch
from loguru import logger
from sklearn.neighbors import KDTree
from torch_geometric.nn import voxel_grid
from torch_geometric.nn import radius_graph
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos
from torch_geometric.nn.pool.pool import pool_batch
from dataset.data import MultiScaleBatch
from dataset.data import MultiScaleData


from torch_radius_search import radius_search


# preprocessing Inputs (creation of the graph)
class DataSubsampling(object):
    """
    precompute subsample versions on the point cloud to perform pooling
    """

    def __init__(self, list_voxel_size):
        """
        perform grid subsampling on point cloud
        list_voxel_size (list of float): list of each grid of pooling
        """
        self.list_voxel_size = list_voxel_size

    def __call__(self, data):

        try:
            batch = data.batch
        except AttributeError:
            batch = torch.zeros(data.x.shape[0])

        # points = [data.pos]
        # First downsample

        points = [data.pos]
        list_pool = []
        list_batch = [batch]
        for ind, voxel_size in enumerate(self.list_voxel_size):

            pool = voxel_grid(points[-1], list_batch[-1], voxel_size)
            pool, perm = consecutive_cluster(pool)
            list_batch.append(pool_batch(perm, list_batch[-1]))
            points.append(pool_pos(pool, points[-1]))

        try:
            res = MultiScaleBatch(
                batch=data.batch,
                list_pool=list_pool,
                points=points,
                list_batch=list_batch,
                x=data.x,
                y=data.y,
                pos=data.pos)
        except AttributeError:
            res = MultiScaleData(
                list_pool=list_pool,
                points=points,
                x=data.x,
                y=data.y,
                pos=data.pos)

        return res


class Neighbors(object):
    """
    compute at each scale, the graph of neighbors (we use a radius neighbors)
    """

    def __init__(self, list_radius, max_num_neighbors=256,
                 is_pool=False, is_upsample=False, mode=0):
        """
        list_radius(list): list of float
        max_num_neighbors(int): maximum number of neighbor we take
        is_pool (boolean): compute the neighbors for the graph pooling
        is_upsample (boolean): compute the neighbors for the upsample
        mode: how do we store the nearest neighbors
        0: list of neighbors(-1 if it is empty)
        1: list of edges (No shadow neighbors) size M x 2
        """
        self.list_radius = list_radius
        self.max_num_neighbors = max_num_neighbors
        self.is_pool = is_pool
        self.is_upsample = is_upsample
        self.mode = mode

    def compute_graph_old(self, query_points, support_points, radius):
        tree = KDTree(support_points.numpy())
        index = list(tree.query_radius(query_points.numpy(), r=radius))
        col = [torch.from_numpy(c).to(torch.long) for c in index]
        row = [torch.full_like(c, i) for i, c in enumerate(col)]
        row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)

        return torch.stack([row, col], dim=0)
        # return index

    def compute_graph(self, query_points, support_points, radius):

        return radius_search(query_points,
                             support_points,
                             radius, self.max_num_neighbors, self.mode)[0]

    def __call__(self, data):
        # TODO batch version
        try:
            list_neigh = []
            list_neigh_size = torch.zeros(len(data.points), dtype=torch.long)

            list_pool = []
            list_pool_size = torch.zeros(len(data.points)-1,
                                         dtype=torch.long)

            list_upsample = []
            list_upsample_size = torch.zeros(len(data.points)-1,
                                             dtype=torch.long)

            for i in range(len(data.points)):

                neigh_index = self.compute_graph(
                    query_points=data.points[i],
                    support_points=data.points[i],
                    radius=self.list_radius[i])
                list_neigh_size[i] = data.points[i].shape[0]
                list_neigh.append(neigh_index)

                if(self.is_pool and i < len(data.points) - 1):
                    pool_index = self.compute_graph(
                        query_points=data.points[i+1],
                        support_points=data.points[i],
                        radius=self.list_radius[i])
                    list_pool_size[i] = data.points[i].shape[0]
                    list_pool.append(pool_index)

                if(self.is_upsample and i < len(data.points) - 1):
                    upsample_index = self.compute_graph(
                        query_points=data.points[i],
                        support_points=data.points[i+1],
                        radius=self.list_radius[i+1])
                    list_upsample_size[i] = data.points[i+1].shape[0]
                    list_upsample.append(upsample_index)

            res = MultiScaleData(
                points=data.points,
                x=data.x,
                y=data.y,
                pos=data.pos,
                list_neigh=list_neigh,
                list_neigh_size=list_neigh_size,
                list_pool=list_pool,
                list_pool_size=list_pool_size,
                list_upsample=list_upsample,
                list_upsample_size=list_upsample_size)

            return res
        except AttributeError:
            logger.error("Error ! We return the input.")
            logger.error(data.keys)
            return data

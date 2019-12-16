# functions to change how we store edges (sparse adjacency matrix, neighborhood matrix, or list of edges)

import torch

def list_neigh2list_edge(list_neigh):
    """
    convert a neighborhood matrix to a list of edges

    inputs:
    list_neigh (torch Tensor long): a N x M matrix where N is the number of points and M is the number of neighbors for each points (-1 if there is no neighbors)

returns:
a list of edges (torch Tensor long of size NM x 2)
    """

    n, m = list_neigh.shape

    node = torch.arange(n, dtype=torch.long).repeat(m, 1).transpose(1, 0).reshape(-1, 1)

    list_edge = torch.cat((node, list_neigh.reshape(-1, 1)), axis=1)
    ind = torch.where((list_edge[:, 0] >= 0) * (list_edge[:, 1] >= 0))[0]

    return list_edge[ind]

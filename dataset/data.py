import re
import torch
import torch_geometric
from torch_geometric.data import Data, Batch


class MultiScaleData(Data):
    r"""A plain old python object modeling a singlemulti scale graph with various
    (optional) attributes:

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        norm (Tensor, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (LongTensor, optional): Face adjacency matrix with shape
            :obj:`[3, num_faces]`. (default: :obj:`None`)

    The data object is not restricted to these attributes and can be extented
    by any other additional data.

    Modify the apply: Now we can apply a function in a list of Tensors
    """

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):

        Data.__init__(self, x=x, edge_index=edge_index,
                      edge_attr=edge_attr,
                      y=y, pos=pos,
                      norm=norm, face=face,
                      **kwargs)

    def apply(self, func, *keys):
        for key, item in self(*keys):
            if(torch.is_tensor(item)):
                self[key] = func(item)
            if(isinstance(item, list)):
                for i in range(len(item)):
                    if(torch.is_tensor(item[i])):
                        self[key][i] = func(item[i])

        return self


class MultiScaleBatch(MultiScaleData, Batch):

    def __init__(self, batch=None, **kwargs):
        MultiScaleData.__init__(self, **kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = MultiScaleBatch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        cumsum4list = {key: [] for key in keys}
        batch.batch = []
        batch.list_batch = []
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))

                # Here particular case for this kind of list of tensors
                # process the neighbors
                if bool(re.search('(neigh|pool|upsample)', key)):
                    if isinstance(item, list):
                        if(len(cumsum4list[key]) == 0): # for the first time
                            cumsum4list[key] = torch.zeros(len(item), dtype=torch.long)
                        for j in range(len(item)):
                            if(torch.is_tensor(item[j]) and item[j].dtype != torch.bool):
                                item[j][item[j] > 0] += cumsum4list[key][j]
                                # print(key, data["{}_size".format(key)][j])
                                cumsum4list[key][j] += data["{}_size".format(key)][j]

                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                cumsum[key] += data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size, ), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

            # indice of the batch at each scale
            if(data.points is not None):
                list_batch = []
                for j in range(len(data['points'])):
                    size = len(data.points[j])
                    item = torch.full((size, ), i, dtype=torch.long)
                    list_batch.append(item)
                batch.list_batch.append(list_batch)


        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:

            item = batch[key][0]

            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])

            elif isinstance(item, list):
                item = batch[key]
                res = []
                for j in range(len(item[0])):
                    col = [f[j] for f in batch[key]]
                    res.append(torch.cat(col,
                                         dim=data_list[0].__cat_dim__(key, col)))
                batch[key] = res
                # print('item', item)

            else:
                raise ValueError('{} is an Unsupported attribute type'.format(type(item)))

        # Copy custom data functions to batch (does not work yet):
        # if data_list.__class__ != Data:
        #     org_funcs = set(Data.__dict__.keys())
        #     funcs = set(data_list[0].__class__.__dict__.keys())
        #     batch.__custom_funcs__ = funcs.difference(org_funcs)
        #     for func in funcs.difference(org_funcs):
        #         setattr(batch, func, getattr(data_list[0], func))

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

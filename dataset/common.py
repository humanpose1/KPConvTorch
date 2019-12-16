import torch
import torch_geometric
from torch.geometry.data import Data


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
    """

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):

        Data.__init__(x, edge_index, edge_attr, y, pos, norm, face, **kwargs)

    def apply(self, func, *keys):
        for key, item in self(*keys):
            if(torch.is_tensor(item)):
                self[key] = func(item)
            if(isinstance(item, list)):
                for i in range(len(item)):
                    if(torch.is_tensor(item[i])):
                        self[key][i] = func(item[i])

        return self

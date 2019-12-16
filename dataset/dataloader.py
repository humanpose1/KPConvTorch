import torch

from dataset.data import MultiScaleBatch
from dataset.data import MultiScaleData


class MultiScaleDataLoader(torch.utils.data.DataLoader):
    r"""
    data loader but the collate function comes from MultiScaleBatch.
    simplified version of
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataloader.html#DataLoader
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        def collate(batch):
            elem = batch[0]
            if(isinstance(elem, MultiScaleData)):
                return MultiScaleBatch.from_data_list(batch, follow_batch)
            else:
                raise TypeError(
                    "the type {} is "
                    "not supported by this dataloader".format(type(elem)))

        super(MultiScaleDataLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=lambda batch: collate(batch), **kwargs)

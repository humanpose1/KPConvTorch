# Now we are ready to train on ModelNet40

from loguru import logger

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.transforms import Compose

from dataset.modelnet import ModelNet, load_subsampled_clouds
from dataset.transform import DataSubsampling, Neighbors
from dataset.dataloader import MultiScaleDataLoader

from models.KPCNN import KPCNN
from models.losses import compute_classification_loss
from models.losses import compute_classification_accuracy
from utility.config import get_cfg_defaults
from utility.misc import get_list_constants, shadow_neigh


def train():

    logger.info("test Dataloader")
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./yaml/modelnet.yaml")

    writer = SummaryWriter()

    v = cfg.NETWORK.FIRST_SUBSAMPLING_DL
    mode_neigh = 0  # neighbors or edge
    architecture, list_voxel_size, list_radius, \
        list_radius_conv, list_size = get_list_constants(
            cfg.NETWORK.FIRST_SUBSAMPLING_DL, cfg.NETWORK.DENSITY_PARAMETER/2,
            cfg.NETWORK.ARCHITECTURE, cfg.NETWORK.FIRST_DIM,
            cfg.INPUT.IN_FEATURES_DIM,
            cfg.INPUT.NUM_CLASSES)

    list_transfo = [DataSubsampling(list_voxel_size)]
    neigh = Neighbors(list_radius,
                      max_num_neighbors=cfg.INPUT.MAX_NUM_NEIGHBORS,
                      is_pool=True, is_upsample=False, mode=mode_neigh)
    list_transfo.append(neigh)

    transfo = Compose(list_transfo)
    # transfo = DataSubsampling(list_voxel_size)

    dataset = ModelNet("./Data/ModelNet40", subsampling_parameter=v,
                       transforms=transfo)

    dataloader = MultiScaleDataLoader(dataset,
                                      batch_size=cfg.NETWORK.BATCH_NUM,
                                      shuffle=True,
                                      num_workers=cfg.SYSTEM.NUM_WORKERS,
                                      pin_memory=False)

    model = KPCNN(architecture, list_radius_conv, list_size, cfg,
                  mode=mode_neigh)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.TRAIN.LEARNING_RATE,
                                momentum=cfg.TRAIN.MOMENTUM)
    model.cuda()
    for e in range(1):
        logger.info("epoch {:d}", e)
        model.train()
        for i, batch in enumerate(dataloader):
            if(mode_neigh == 0):
                batch = shadow_neigh(batch).to('cuda')
            else:
                batch = batch.to('cuda')
            # print(batch.list_neigh[3][:20])
            pred = model(batch)
            loss = compute_classification_loss(pred,
                                               batch.y,
                                               model,
                                               cfg.TRAIN.WEIGHTS_DECAY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(i % cfg.TRAIN.LOG_INTERVAL == 0):
                accuracy = compute_classification_accuracy(pred, batch.y)
                global_step = e*len(dataloader)+i
                logger.info("Epoch: {} Step {}, loss {:3f}, accuracy: {:3f}",
                            e, i, loss.item(), accuracy.item())
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Accuracy/train', accuracy.item(),
                                  global_step)


if __name__ == '__main__':
    train()

# Blocks ResNet, Simple Upsample  pooling...
# it always take a data or batch as input and modify the x attribute
# (ie the features)
# TODO : Upsampling layers for segmentations
#

import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import max_pool
from kernels.kp_module import PointKernel
from models.layers import KPConvLayer
from models.layers import DeformableKPConvLayer
from models.layers import UnaryConv
from models.layers import max_pool


class Block(torch.nn.Module):
    """
    basic block
    method:
    - activate_feature: BN + activation
    """
    def __init__(self):
        super(Block, self).__init__()

    def forward(self, data):
        raise NotImplementedError("This is an abstract basic Block")

    def activate_feature(self, x, bn):
        """
        batch norm and activation function.
        """
        if(self.config.NETWORK.USE_BATCH_NORM):
            return self.activation(bn(x))
        else:
            return self.activation(x)


class SimpleBlock(Block):
    """
    simple layer with KPConv convolution -> activation -> BN
    we can perform a stride version (just change the query and the neighbors)
    """

    def __init__(self, num_inputs, num_outputs, layer_ind,
                 kp_conv,
                 config,
                 is_strided=False,
                 activation=torch.nn.LeakyReLU(negative_slope=0.2)):

        super(SimpleBlock, self).__init__()
        self.layer_ind = layer_ind
        self.config = config
        self.is_strided = is_strided
        self.kp_conv = kp_conv
        self.bn = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)
        self.activation = activation

    def forward(self, data):

        inputs = data.x
        if(not self.is_strided):
            x = self.kp_conv(
                pos=(data.points[self.layer_ind],
                     data.points[self.layer_ind]),
                neighbors=data.list_neigh[self.layer_ind],
                x=inputs)
        else:
            x = self.kp_conv(
                pos=(data.points[self.layer_ind],
                     data.points[self.layer_ind+1]),
                neighbors=data.list_pool[self.layer_ind],
                x=inputs)
        x = self.activate_feature(x, self.bn)
        data.x = x
        return data


class UnaryBlock(Block):

    """
    layer with  unary convolution -> activation -> BN
    """
    def __init__(self, num_inputs, num_outputs,
                 config, activation=torch.nn.LeakyReLU(negative_slope=0.2)):
        super(UnaryBlock, self).__init__()
        self.uconv = UnaryConv(num_inputs, num_outputs,
                               config)
        self.bn = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)
        self.activation = activation

    def forward(self, data):
        inputs = data.x
        x = self.uconv(inputs)
        x = self.activate_feature(x, self.bn)
        data.x = x
        return data


class ResnetBlock(Block):

    """
    layer with KPConv with residual units
    KPConv -> KPConv + shortcut
    """
    def __init__(self, num_inputs, num_outputs, radius, layer_ind,
                 config,
                 is_strided=False,
                 activation=torch.nn.LeakyReLU(negative_slope=0.2)):
        super(ResnetBlock, self).__init__()
        self.layer_ind = layer_ind
        self.is_strided = is_strided
        self.config = config

        self.size = [num_inputs, num_outputs, num_outputs, num_outputs]
        self.kp_conv0 = KPConvLayer(radius, num_inputs, num_outputs, config)
        self.kp_conv1 = KPConvLayer(radius, num_outputs, num_outputs, config)

        self.activation = activation
        self.bn0 = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        self.bn1 = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        if(num_inputs != num_outputs):
            self.shortcut_op = UnaryConv(num_inputs, num_outputs,
                                         config)
        else:
            self.shortcut_op = torch.nn.Identity()

    def forward(self, data):
        inputs = data.x
        x = self.kp_conv0(data.points[self.layer_ind],
                          data.points[self.layer_ind],
                          data.list_neigh[self.layer_ind],
                          inputs)

        x = self.activate_feature(x, self.bn0)
        if(not self.is_strided):
            x = self.kp_conv1(data.points[self.layer_ind],
                              data.points[self.layer_ind],
                              data.list_neigh[self.layer_ind],
                              x)
            x = self.activate_feature(x, self.bn1)
            data.x = x + self.shortcut_op(inputs)

        else:
            x = self.kp_conv(data.points[self.layer_ind+1],
                             data.points[self.layer_ind],
                             data.list_pool[self.layer_ind],
                             x)
            x = self.activate_feature(x, self.bn1)
            shortcut = self.shortcut_op(max_pool(
                inputs, data.list_pool[self.layer_ind]))
            data.x = x + shortcut

        return data


class ResnetBottleNeckBlock(Block):
    """
    uconv -> kpconv -> uconv + shortcut
    """
    def __init__(self, num_inputs, num_outputs, layer_ind,
                 kp_conv,
                 config,
                 is_strided=False,
                 activation=torch.nn.LeakyReLU(negative_slope=0.2)):
        super(ResnetBottleNeckBlock, self).__init__()
        self.config = config
        self.layer_ind = layer_ind
        self.is_strided = is_strided

        self.uconv0 = UnaryConv(num_inputs, num_outputs//4, config)
        # self.kp_conv = KPConvLayer(radius, self.size[1], self.size[2], config)
        self.kp_conv = kp_conv
        self.uconv1 = UnaryConv(num_outputs//4, num_outputs, config)

        self.activation = activation
        self.bn0 = torch.nn.BatchNorm1d(
            num_outputs//4, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        self.bn1 = torch.nn.BatchNorm1d(
            num_outputs//4, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        self.bn2 = torch.nn.BatchNorm1d(
            num_outputs, momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        if(num_inputs != num_outputs):
            self.shortcut_op = UnaryConv(num_inputs, num_outputs,
                                         config)
        else:
            self.shortcut_op = torch.nn.Identity()

    def forward(self, data):

        inputs = data.x
        x = self.uconv0(inputs)
        x = self.activate_feature(x, self.bn0)

        if(not self.is_strided):
            x = self.kp_conv(pos=(data.points[self.layer_ind],
                                  data.points[self.layer_ind]),
                             neighbors=data.list_neigh[self.layer_ind],
                             x=x)
            x = self.activate_feature(x, self.bn1)
        else:
            x = self.kp_conv(pos=(
                data.points[self.layer_ind],
                data.points[self.layer_ind+1]),
                             neighbors=data.list_pool[self.layer_ind],
                             x=x)
            x = self.activate_feature(x, self.bn1)
        x = self.uconv1(x)
        x = self.activate_feature(x, self.bn2)
        if(not self.is_strided):
            data.x = x + self.shortcut_op(inputs)
        else:
            data.x = x + self.shortcut_op(
                max_pool(inputs, data.list_pool[self.layer_ind]))
        return data


class MaxPool(torch.nn.Module):
    """
    layer that perform max_pooling
    TODO : when the neural network is
    """
    def __init__(self, layer_ind):
        super(MaxPool, self).__init__()
        self.layer_ind = layer_ind

    def forward(self, data):
        inputs = data.x
        if(data.pools[self.layer_ind].shape[1] > 2):
            x = max_pool(inputs, data.pools[self.layer_ind])
        else:
            raise NotImplementedError("implement for list of edges")
            x = None
        data.x = x
        return data


class GlobalPool(torch.nn.Module):
    """
    global pooling layer

    mode: 0 use torch scatter to pool the point cloud
    mode 1 use for loop
    """
    def __init__(self, mode=0):
        self.mode = mode
        super(GlobalPool, self).__init__()

    def forward(self, data):

        inputs = data.x
        batch = data.list_batch[-1]
        if(len(inputs) != len(batch)):
            raise Exception("Error, the batch and the features have not the same size")

        if(self.mode == 0):
            x = global_mean_pool(inputs, batch)
        elif(self.mode == 1):
            ind_batch = torch.unique(batch)
            list_pool = [inputs[batch == ind].mean(axis=0) for ind in ind_batch]
            x = torch.stack(list_pool)

        data.x = x
        return data


class NearestUpsample(Block):

    def __init__(self, layer_ind):
        super(NearestUpsample, self).__init__()
        self.layer_ind = layer_ind

    def forward(self, data):
        if(data.list_upsample[self.layer_ind].size(1) > 2):
            shadow = torch.zeros(1, data.x.size(1),
                                 dtype=data.x.dtype,
                                 device=data.x.device)
            inputs = torch.cat([data.x, shadow],
                               axis=0)
            inds = data.list_upsample[self.layer_ind][:, 0]
            res = inputs[inds]
            data.x = res
            return data

        else:
            inputs = data.x
            col, row = data.list_upsample[self.layer_ind].t()
            shortest = torch.cumsum(row.bincount(), 0)
            shortest[-1] = 0
            inds = col[shortest]
            res = inputs[inds]
            data.x = res
            return data


class MLPClassifier(Block):
    """
    two layer of MLP multi class classification

    """

    def __init__(self, num_inputs, num_classes, config,
                 num_hidden=1024,
                 dropout_prob=0.5,
                 activation=torch.nn.LeakyReLU(negative_slope=0.2)):
        super(MLPClassifier, self).__init__()
        self.config = config
        self.lin0 = torch.nn.Linear(num_inputs, num_hidden)
        self.bn0 = torch.nn.BatchNorm1d(
            num_hidden,
            momentum=self.config.NETWORK.BATCH_NORM_MOMENTUM)

        self.dropout = torch.nn.Dropout(p=dropout_prob)

        self.lin1 = torch.nn.Linear(num_hidden, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.activation = activation

    def forward(self, data):

        inputs = data.x

        x = self.activate_feature(self.lin0(inputs), self.bn0)
        x = self.dropout(x)
        x = self.lin1(x)
        x = self.softmax(x)
        data.x = x
        return data

    # the block ops

# TODO segmenter module


def get_block_ops(name, num_inputs, num_outputs, layer_ind,
                  radius, config, mode=0):
    """
    a small function that give the block operation
    inputs:
    - name: name of the block
    - num_inputs: dimension of the inputs
    - num_outputs: dimension of the outputs
    - layer_ind: which layer it is(too choose the right support point)
    - radius: radius for the convolution
    - config: YACS object containing all constants and hyper parameters
    - mode: 0 the convolution with shadow neighbors mode 1 convolution using message passing
    return:
    - a torch Module
    TODO : upsample blocks for segmentation
    """

    if("simple" in name):
        kp_conv = None
        if(mode == 0):
            kp_conv = KPConvLayer(radius, num_inputs, num_outputs, config)
        elif(mode == 1):
            kp_conv = PointKernel(config.NETWORK.NUM_KERNEL_POINTS,
                                  num_inputs, num_outputs, radius=radius)
    elif("resnetb" in name):
        kp_conv = None
        if(mode == 0):
            kp_conv = KPConvLayer(radius, num_outputs//4, num_outputs//4,
                                  config)
        elif(mode == 1):
            kp_conv = PointKernel(config.NETWORK.NUM_KERNEL_POINTS,
                                  num_outputs//4, num_outputs//4,
                                  radius=radius)

    if(name == "simple"):
        block = SimpleBlock(num_inputs, num_outputs, layer_ind,
                            kp_conv, config)
    elif(name == "unary"):
        block = UnaryBlock(num_inputs, num_outputs,
                           config)
    elif(name == "simple_strided"):
        block = SimpleBlock(num_inputs, num_outputs, layer_ind, kp_conv,
                            config,
                            is_strided=True)
    elif(name == "resnet"):
        block = ResnetBlock(num_inputs, num_outputs, layer_ind, config)

    elif(name == "resnetb"):

        block = ResnetBottleNeckBlock(num_inputs, num_outputs,
                                      layer_ind, kp_conv, config)

    elif name == "resnetb_strided":

        block = ResnetBottleNeckBlock(num_inputs, num_outputs,
                                      layer_ind, kp_conv, config,
                                      is_strided=True)

    elif name == "max_pool":
        block = MaxPool(layer_ind)
    elif name == "global_average":
        block = GlobalPool(mode=0)
    else:
        raise ValueError("Unknown block name in the architecture definition : " + name)
    return block


def assemble_block(list_name, list_radius, list_size, config, mode):
    """
    assemble block
    parameters:
    list_name (list of String of size N): the list of the name of each block
    list_radius(list of float of size N): list of radius for each block
    list_size (list of tuple of int): the size of the input and the output
    config: config File for the hyperparameters and constants
    return :
    a torch.nn.Module block.
    """
    list_block = []
    layer_ind = 0
    for i, name in enumerate(list_name):

        block = get_block_ops(name, list_size[i][0], list_size[i][1],
                              layer_ind, list_radius[i], config, mode=mode)
        list_block.append(block)
        if("strided" in name or "max_pool" in name):
            layer_ind += 1
    return torch.nn.Sequential(*list_block)

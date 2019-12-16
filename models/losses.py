# Losses

import torch
import torch.nn.functional as F


def compute_regularisation_loss(model, is_cuda=True):
    """
    the classical L2 loss on weights
    """

    L_reg = 0
    for name, param in model.named_parameters():
        if('weight' in name):

            L_reg = L_reg + torch.norm(param, 2)

    return L_reg


def compute_deformable_regularisation_loss(model):
    """
    regularisation on deformable convolution
    TODO
    """
    return torch.tensor(0., requires_grad=True)


def compute_classification_loss(prediction, labels, model, weight_decay):
    cls_loss = F.nll_loss(prediction, labels)
    reg_loss = compute_regularisation_loss(model)

    return cls_loss + weight_decay * reg_loss


def compute_segmentation_loss(prediction, labels, model, weight_decay):
    pass


def compute_batch_hard_triplet_loss():
    pass


def compute_classification_accuracy(prediction, labels):
    _, argmax = torch.max(prediction, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()
    return accuracy

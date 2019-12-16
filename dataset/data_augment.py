# Data augmentation class
# TODO Translation Rotation Symmetry Noise Scale
# before multiscaling and neighbors

import torch


class RandomTranslate(object):

    def __init__(self, translate):
        self.translate = translate

    def __call__(self, data):

        t = 2*(torch.rand(3)-0.5)*self.translate

        data.pos = data.pos + t
        return data

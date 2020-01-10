# Data augmentation class
# TODO Translation Rotation Symmetry Noise Scale
# before multiscaling and neighbors
# TODO what about normals

import math
import torch


class Center(object):

    def __call__(self, data):
        data.pos = data.pos - data.pos.mean(axis=0)
        return data

    def __repr__(self):
        return "Center"


class RandomTranslate(object):

    def __init__(self, translate):
        self.translate = translate

    def __call__(self, data):

        t = 2*(torch.rand(3)-0.5)*self.translate

        data.pos = data.pos + t
        return data

    def __repr__(self):
        return "Random Translate of translation {}".format(self.translate)


class RandomScale(object):

    def __init__(self, scale_min=1, scale_max=1, is_anisotropic=False):
        if(scale_min > scale_max):
            raise ValueError("Scale min must be lesser or equal to Scale max")
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.is_anisotropic = is_anisotropic

    def __call__(self, data):
        scale = self.scale_min +\
            torch.rand(1) * (self.scale_max - self.scale_min)
        if(self.is_anisotropic):
            ax = torch.randint(0, 3, 1)
            data.pos[:, ax] = scale * data.pos[:, ax]
        else:
            data.pos = scale * data.pos
        return data

    def __repr__(self):
        return "Random Scale min={}, max={}".format(self.scale_min,
                                                    self.scale_max)


class RandomSymmetry(object):

    def __init__(self, axis=[False, False, False]):
        self.axis = axis

    def __call__(self, data):

        for i, ax in enumerate(self.axis):
            if(ax):
                if(torch.rand(1) < 0.5):
                    data.pos[:, i] *= -1
        return data

    def __repr__(self):
        return "Random symmetry of axes: x={}, y={}, z={}".format(*self.axis)


class RandomNoise(object):

    def __init__(self, sigma=0.0001):
        """
        simple isotropic additive gaussian noise
        """
        self.sigma = sigma

    def __call__(self, data):

        noise = self.sigma * torch.randn(data.pos.shape)
        data.pos = data.pos + noise
        return data

    def __repr__(self):
        return "Random noise of sigma={}".format(self.sigma)


def euler_angles_to_rotation_matrix(theta):
    """

    """
    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta[0]), -torch.sin(theta[0])],
                        [0, torch.sin(theta[0]), torch.cos(theta[0])]])

    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])],
                        [0, 1, 0],
                        [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])

    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0],
                        [torch.sin(theta[2]), torch.cos(theta[2]), 0],
                        [0, 0, 1]])

    R = torch.mm(R_z, torch.mm(R_y, R_x))
    return R


class RandomRotation(object):

    def __init__(self, mode='vertical'):
        """
        random rotation: either
        """
        self.mode = mode

    def __call__(self, data):

        theta = torch.zeros(3)
        if(self.mode == 'vertical'):
            theta[2] = torch.rand(1) * 2 * torch.tensor(math.pi)
        elif(self.mode == 'all'):
            theta = torch.rand(3) * 2 * torch.tensor(math.pi)
        else:
            raise NotImplementedError("this kind of rotation ({}) "
                                      "is not yet available".format(self.mode))
        R = euler_angles_to_rotation_matrix(theta)
        data.pos = torch.mm(data.pos, R.t())
        return data

    def __repr__(self):
        return "Random rotation of mode {}".format(self.mode)

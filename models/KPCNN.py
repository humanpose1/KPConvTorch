# Classifier
import torch
from models.blocks import MLPClassifier
from models.blocks import assemble_block


class KPCNN(torch.nn.Module):
    """
    classifier
    it contains two main parts:
    an encoder part (using kernel point convolution) that performs embedding.
    classifier : an classical MLP
    """

    def __init__(self, list_name, list_radius, list_size, config, mode=0):
        super(KPCNN, self).__init__()

        self.encoder = assemble_block(list_name, list_radius, list_size,
                                      config, mode=mode)
        self.classifier = MLPClassifier(list_size[-1][0], list_size[-1][1],
                                        config)

    def forward(self, data):
        dat = data
        dat = self.encoder(dat)
        dat = self.classifier(dat)
        return data.x

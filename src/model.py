import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models.vgg
from math import sqrt


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") - only the conv part

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    pass

anchors_sizes = [128, 256, 512]
anchor_ratios = [1, 2, 0.5]

def generate_anchors():
    for size in anchors_sizes:
        for ratio in anchor_ratios:
            w = sqrt((size ** 2) / ratio)
            h = (size ** 2) / w
            w = int(w)
            h = int(h)
            yield (w,h,)

anchors = list(generate_anchors())

'''
The region proposal network
it is attached on top of backbone network.
It outputs anchor box position and probabilities feature maps
'''
class RPNHead(nn.Module):
    def __init__(self, input_features = 512, n = 3):
        super(RPNHead, self).__init__()

        self.k = len(anchors)
        self.network = self.make_network(input_features, n = n, k = self.k)

    def make_network(self,input_features_count, n, k):
        cnet = nn.Conv2d(input_features_count, input_features_count, n)

        # Position and probabilities feature maps
        pfc = nn.Conv2d(input_features_count, 6 * k, 1)

        return nn.Sequential([
            nn.ReLU(True),
            cnet,
            nn.ReLU(True),
            pfc
        ])

    def forward(self, x):
        return x


import torch
import torch.nn as nn
import torch.functional as F
from torchvision.models.vgg import VGG, make_layers, model_urls
import torch.utils.model_zoo as model_zoo
from math import sqrt
import math


def vgg16(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D") - only the conv part

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    conf = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(conf), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))

    # Froze bottom layers
    for module in model.features[0:10]:
        for param in module.parameters():
            param.requires_grad = False

    return model.features

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
        self._initialize_weights()

    def make_network(self,input_features_count, n, k):
        cnet = nn.Conv2d(input_features_count, input_features_count, n)

        # Position and probabilities feature maps
        pfc = nn.Conv2d(input_features_count, 6 * k, 1)

        return nn.Sequential(*[
            cnet,
            nn.ReLU(True),
            pfc
        ])

    def forward(self, x):
        return self.network(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


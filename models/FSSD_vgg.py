import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models import vgg, vgg_base


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None  # nn.Upsample --> nn.functional.interpolate
                                                                                                                        # mode='bilinear' --> align_corners=False
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class FSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1712.00960.pdf or more details.

    Args:
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, vgg, extras, ff_layers, pyramid_ext, head, num_classes, size):
        super(FSSD, self).__init__()
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(vgg)             # From list to nn.ModuleList!!!. (list is consist of Custom nn.Module layers)
        self.extras = nn.ModuleList(extras)
        self.ff_layers = nn.ModuleList(ff_layers) # It is just 3 convolution layers!!!

        self.pyramid_ext = nn.ModuleList(pyramid_ext)
        self.fea_bn = nn.BatchNorm2d(256 * len(self.ff_layers), affine=True)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax()

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        source_features = list()
        transformed_features = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(22):
            x = self.vgg[k](x)

        source_features.append(x)

        # apply vgg up to fc7
        for k in range(22, len(self.vgg)):
            x = self.vgg[k](x)
        source_features.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)

        source_features.append(x)
        assert len(self.ff_layers) == len(source_features) # Is it the same size that after 'size(3 feature maps) == size(3 filters)'.
        for i, conv in enumerate(self.ff_layers):
            transformed_features.append(conv(source_features[i]))

        concat_fea = torch.cat(transformed_features, 1)
        #print('concat_fea : ', concat_fea.data) #([8, 256*3, 38,38])
        x = self.fea_bn(concat_fea) # plz refer paper.

        pyramid_fea = list()
        for i, conv in enumerate(self.pyramid_ext):
            x = conv(x)
            pyramid_fea.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(pyramid_fea, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if test:
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def feature_transform_module(vgg, extras, size):
    if size == 300:
        up_size = 38
    elif size == 512:
        up_size = 64

    ff_layers = []
    # conv4_3
    ff_layers += [BasicConv(vgg[21].out_channels, 256, kernel_size=1, padding=0)] #24 -> 21
    #print('vgg[24].out_channels: ',vgg[24].out_channels, 'vgg[24].kernel_size: ',vgg[24].kernel_size)
    # fc_7
    ff_layers += [BasicConv(vgg[-2].out_channels, 256, kernel_size=1, padding=0, up_size=up_size)]
    #print('vgg[-2].out_channels: ', vgg[-2].out_channels, 'vgg[-2].kernel_size: ',vgg[-2].kernel_size)
    # conv8_2
    ff_layers += [BasicConv(extras[-1].out_channels, 256, kernel_size=1, padding=0, up_size=up_size)]
    #print('extral[-1].out_channels: ',extras[-1].out_channels)
    return vgg, extras, ff_layers

extras = {
    '300': [256, 512, 128, 'S', 256],
    '512': [256, 512, 128, 'S', 256],
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}
print('vgg : \n')
for idx, i in enumerate(vgg(vgg_base['300'], 3)):
    print(idx, i)
print('add_extras : \n')
for idx, i in enumerate(add_extras(extras['300'], 1024)):
    print(idx," ", i)

# original_extras = {
#     '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#     '512': [],
# }
# original_mbox = {
#     '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     '512': [],
# }

def pyramid_feature_extractor(size):
    if size == 300:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0),
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0)]
    elif size == 512:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
    return layers


def multibox(fea_channels, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    assert len(fea_channels) == len(cfg)
    for i, fea_channel in enumerate(fea_channels):
        loc_layers += [nn.Conv2d(fea_channel, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


fea_channels = {
    '300': [512, 512, 256, 256, 256, 256],
    '512': [512, 512, 256, 256, 256, 256, 256]}

def build_net(size=300, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only FSSD300 and FSSD512 is supported currently!")
        return

    return FSSD(*feature_transform_module(vgg(vgg_base[str(size)], 3), add_extras(extras[str(size)], 1024), size=size),
                pyramid_ext=pyramid_feature_extractor(size),
                head=multibox(fea_channels[str(size)], mbox[str(size)], num_classes), num_classes=num_classes,
                size=size)

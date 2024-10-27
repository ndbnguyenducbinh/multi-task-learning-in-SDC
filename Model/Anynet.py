import torch 
import torch.nn as nn
import torch.functional as F
from math import sqrt

# Stem
class Stem(nn.Module):

    def __init__(self, out_channels ):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        return x
    

#Body (include Stage, and each Stage include Block)
class XBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width, stride):
        super(XBlock, self).__init__()
        bottleneck_channels = in_channels // bottleneck_ratio
        groups = bottleneck_channels // group_width

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU()
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU()
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if stride !=1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
        
        self.rl = nn.ReLU()
        
    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        if self.shortcut is not None:
            x2 = self.shortcut(x)
        else:
            x2 = x
        x = self.rl(out + x2)
        return x

class Stage(nn.Module):

    def __init__(self,num_blocks, in_channels, out_channels, bottleneck_ratio, group_width, stride):
        super(Stage, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module("block_0", XBlock(in_channels, out_channels, bottleneck_ratio, group_width, stride))
        for i in range(1, num_blocks):
            self.blocks.add_module("block{}".format(i), 
                                   XBlock(out_channels, out_channels, bottleneck_ratio, group_width, stride=1))
            
    def forward(self, x):
        x = self.blocks(x)
        return x
    

class Any_net_X(nn.Module):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride):
        super(Any_net_X, self).__init__()

        for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0

        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module("stem", Stem(prev_block_width))

        self.net.add_module("stage_0", nn.Sequential(
            nn.Conv2d(prev_block_width, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            XBlock(64, 64, bottleneck_ratio=1, group_width=16, stride=2)
        ))

        self.net.add_module("stage_1", nn.Sequential(
            XBlock(64, 128, bottleneck_ratio=1, group_width=16, stride=2),
            XBlock(128, 128, bottleneck_ratio=1, group_width=16, stride=1)
        ))

        self.net.add_module("stage_2", nn.Sequential(
            XBlock(128, 256, bottleneck_ratio=1, group_width=16, stride=2),
            XBlock(256, 256, bottleneck_ratio=1, group_width=16, stride=1)
        ))

        self.net.add_module("stage_3", nn.Sequential(
            XBlock(256, 512, bottleneck_ratio=1, group_width=16, stride=2),
            XBlock(512, 512, bottleneck_ratio=1, group_width=16, stride=1)
        ))

        self.initialize_weight()
        self.stage_num = 4

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net[0](x)

        feat = list()
        for i in range(self.stage_num):
            x = self.net[1 + i](x)
            feat.append(x)
        return feat
        
        
'''
        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in enumerate(zip(ls_num_blocks, ls_block_width,
                                                                                         ls_bottleneck_ratio,
                                                                                         ls_group_width)):
            self.net.add_module("stage_{}".format(i),
                               Stage(num_blocks,
                                     prev_block_width,
                                     block_width,
                                     bottleneck_ratio,
                                     group_width,
                                     stride))
            prev_block_width = block_width
        
        self.initialize_weight()
        self.stage_num = len(self.net) - 1

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net[0](x)

        feat = list()
        for i in range(self.stage_num):
            x = self.net[1 + i](x)
            feat.append(x)
        return feat
'''    

class AnyNetXb(Any_net_X):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride):
        super(AnyNetXb, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride)
        assert len(set(ls_bottleneck_ratio)) == 1


class AnyNetXc(Any_net_X):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride):
        super(AnyNetXc, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride)
        assert len(set(ls_group_width)) == 1


class AnyNetXd(Any_net_X):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride):
        super(AnyNetXd, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride)
        assert all(i <= j for i, j in zip(ls_block_width, ls_block_width[1:])) is True


class AnyNetXe(Any_net_X):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride):
        super(AnyNetXe, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride)
        if len(ls_num_blocks > 2):
            assert all(i <= j for i, j in zip(ls_num_blocks[:-2], ls_num_blocks[1:-1])) is True







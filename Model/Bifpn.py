import torch
from torch import nn
from Common import SeparableConvBlock, MaxPool2dStaticSamePadding,MemoryEfficientSwish,Swish,Conv2dStaticSamePadding
import torch.nn.functional as F

"""
BiFPN take input from 4 different layers and output 4 different layers
inculde 160x120x64, 80x60x128, 40x30x256, 20x15x512 from regnet

"""

'''
input parameter:
    num_channels: number of channels in each input tensor
    conv_channels: number of channels in each intermediate tensor
    epsilon: small value to avoid division by zero
    attention: whether to use attention in the fusion
'''

class BiFpn(nn.Module):

    def __init__(self, num_channels, conv_channels,first_time=False, epsilon=1e-4, attention=True):
        super(BiFpn, self).__init__()
        self.epsilon = epsilon
        self.first_time = first_time
        self.attention = attention

        #upsampling layer
        self.conv6_up = SeparableConvBlock(num_channels, num_channels)
        self.conv5_up = SeparableConvBlock(num_channels, num_channels)
        self.conv4_up = SeparableConvBlock(num_channels, num_channels)

        #downsampling layer
        self.conv4_down = SeparableConvBlock(num_channels, num_channels)
        self.conv5_down = SeparableConvBlock(num_channels, num_channels)
        self.conv6_down = SeparableConvBlock(num_channels, num_channels)

        if self.first_time:
            self.p6_down = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        if self.attention:
            self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p6_w1_relu = nn.ReLU()
            self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p5_w1_relu = nn.ReLU()
            self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p4_w1_relu = nn.ReLU()
            self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p3_w1_relu = nn.ReLU()

            self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p4_w2_relu = nn.ReLU()
            self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p5_w2_relu = nn.ReLU()
            self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p6_w2_relu = nn.ReLU()

        self.swish = MemoryEfficientSwish()

    def forward(self, features):
        """
        P6_0 ---------> P6_1 ---------> P6_2 -------->
           |-------------|--------------↑ 
                         ↓                
        P5_0 ---------> P5_1 ---------> P5_2 -------->
           |-------------|--------------↑ 
                         ↓                
        P4_0 ---------> P4_1 ---------> P4_2 -------->
           |-------------|--------------↑ 
                         ↓                
        P3_0 -------------------------- P3_2 -------->
        """

        if self.first_time:
            p3, p4, p5, p6 = features
            p3_in = self.p3_down(p3)
            p4_in = self.p4_down(p4)
            p5_in = self.p5_down(p5)
            p6_in = self.p6_down(p6)
        else:
            p3_in, p4_in, p5_in, p6_in = features

        # Upward path
        if self.attention:
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_up = self.conv6_up(weight[0] * self._upsample(p6_in, p5_in.size()[2:]) + weight[1] * p5_in)

            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * p6_up)

            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self._upsample(p5_up, p4_in.size()[2:]))
        else:
            p6_up = self.conv6_up(self._upsample(p6_in, p5_in.size()[2:]) + p5_in)
            p5_up = self.conv5_up(p5_in + p6_up)
            p4_up = self.conv4_up(self._upsample(p5_up, p4_in.size()[2:]) + p4_in)

        # Downward path
        if self.attention:
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                weight[0] * p4_in + weight[1] * p4_up + weight[2] * self._upsample(p5_up, p4_in.size()[2:])
            )

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                weight[0] * p5_in + weight[1] * p5_up + weight[2] * self._downsample(p4_out, p5_in.size()[2:])
            )

            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                weight[0] * p6_in + weight[1] * p6_up + weight[2] * self._downsample(p5_out, p6_in.size()[2:])
            )
        else:
            p4_out = self.conv4_down(p4_in + p4_up + self._upsample(p5_up, p4_in.size()[2:]))
            p5_out = self.conv5_down(p5_in + p5_up + self._downsample(p4_out, p5_in.size()[2:]))
            p6_out = self.conv6_down(p6_in + p6_up + self._downsample(p5_out, p6_in.size()[2:]))

        return p3_in, p4_out, p5_out, p6_out

    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode='nearest')

    def _downsample(self, x, size):
        return F.adaptive_max_pool2d(x, output_size=size)


class stackBiFpn(nn.Module):
    def __init__(self, fpn_num_filters, fpn_cell_repeats, conv_channels):
        super(stackBiFpn, self).__init__()
        self.fpn_num_filters = fpn_num_filters
        self.fpn_cell_repeats = fpn_cell_repeats
        self.conv_channels = conv_channels
       

        self.bifpn_cells = nn.ModuleList()
        for repeats in range(self.fpn_cell_repeats):
            self.bifpn_cells.append(
                BiFpn(
                    num_channels=self.fpn_num_filters,
                    conv_channels=self.conv_channels, 
                )
            )

    def forward(self, inputs):
        """
        Args:
            inputs (list): A list of tensors [P3, P4, P5, P6], each with shape (batch_size, C, H, W)
        Returns:
            outputs (list): A list of tensors [P3, P4, P5, P6] after BiFPN processing
        """
        outputs = inputs
        for bifpn_cell in self.bifpn_cells:
            outputs = bifpn_cell(outputs)
        return outputs

if __name__ == '__main__':

    # Giả lập các feature maps đầu vào
    feat1 = torch.randn(1, 64, 160, 120).to('cuda')
    feat2 = torch.randn(1, 128, 80, 60).to('cuda')
    feat3 = torch.randn(1, 256, 40, 30).to('cuda')
    feat4 = torch.randn(1, 512, 20, 15).to('cuda')

    features = [feat1, feat2, feat3, feat4]

    # Khởi tạo mô hình stackBiFpn
    model = stackBiFpn(fpn_num_filters=64, fpn_cell_repeats=3, conv_channels=[64, 128, 256, 512]).to('cuda')
    
    # Chạy forward pass
    outputs = model(features)
    
    # In kích thước của các feature maps đầu ra
    for i, output in enumerate(outputs):
        print(f"P{i+3} output shape: {output.shape}")

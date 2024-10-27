import numpy as np
from Anynet import AnyNetXe

class RegNetX(AnyNetXe):
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride):

        parameterized_width = initial_width + slope * np.arange(network_depth)
        parameterized_block = np.log(parameterized_width / initial_width) / np.log(quantized_param)
        parameterized_block = np.round(parameterized_block)
        quantized_width = initial_width * np.power(quantized_param, parameterized_block)

        quantized_width = 8 * np.round(quantized_width / 8)
        ls_block_width, ls_num_blocks = np.unique(quantized_width.astype(np.int32), return_counts=True)

        ls_group_width = np.array([min(group_width, block_width // bottleneck_ratio) for block_width in ls_block_width])
        ls_block_width = np.round(ls_block_width // bottleneck_ratio / group_width) * group_width
        ls_group_width = ls_group_width.astype(np.int32) * bottleneck_ratio
        ls_bottleneck_ratio = [bottleneck_ratio for _ in range(len(ls_block_width))]

        super(RegNetX, self).__init__(ls_num_blocks, ls_block_width.astype(np.int32).tolist(), ls_bottleneck_ratio,
                                       ls_group_width.tolist(), stride)
        
        
class RegNetY(RegNetX):
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride):
        super(RegNetY, self).__init__(initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride)

if __name__ == '__main__':
    bottleneck_ratio = 1
    group_width = 8
    initial_width = 24
    slope = 36
    quantized_param = 2.5
    network_depth = 30
    stride = 2

    model = RegNetY(initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride)

    model.to("cuda:0")
    model.eval()

    # inference
    import torch
    
    dummy_input = torch.randn((1, 3, 1280, 960)).to("cuda:0")
    dummy_output = model(dummy_input)
    print(len(dummy_output))
    for output in dummy_output:
        print(output.shape)
    
        
        
        
        
        
        
        

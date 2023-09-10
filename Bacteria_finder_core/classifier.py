import numpy as np
import torch
import torch.nn as nn
from torch import nn
from types import SimpleNamespace

torch.manual_seed(810)

class MobileNetBlock(nn.Module):

    def __init__(self, c_in, act_fn, c_out, expand_ratio, stride):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        hidden_dim = int(c_in * expand_ratio)
        self.stride = stride
        self.use_res_connect = self.stride == 1 and c_in == c_out

        if expand_ratio == 1:
            self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim, 1*hidden_dim, kernel_size=3, padding=1, stride = stride, bias = False, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            act_fn(),
            nn.Conv2d(hidden_dim, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out),
        )
        else:
            self.net = nn.Sequential(
            nn.Conv2d(c_in, 1*hidden_dim, 1, 1, 0, bias = False),
            nn.BatchNorm2d(hidden_dim),
            act_fn(),
            # dw
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_fn(),
            # linear
            nn.Conv2d(hidden_dim, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out),
        )
        

    def forward(self, x):
        if self.use_res_connect:
            z = x + self.net(x)
        else:
            z = self.net(x)
        return z

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}

resnet_blocks_by_name = {
    "MobileNetBlock": MobileNetBlock
}
    
class MobileNetV2(nn.Module):

    def __init__(self, num_classes=4, num_blocks=[1,2,3,4,3,3,1], expand_ratio = [1,6,6,6,6,6,6],
                c_hidden=[16, 24, 32, 64, 96, 160, 320], strides = [1, 2, 2, 2, 1, 2, 1], width_mult = 1,
                act_fn_name="relu6", block_name="MobileNetBlock", **kwargs):
        """
        Inputs:
            num_classes - Number of classification outputs
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
            block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
        """
        super().__init__()
        assert block_name in resnet_blocks_by_name
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       expand_ratio=expand_ratio,
                                       strides = strides,
                                       c_hidden=c_hidden,
                                       num_blocks=num_blocks,
                                       width_mult=width_mult,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name],
                                       block_class=resnet_blocks_by_name[block_name])
        self.input_channel = 32
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        strides = self.hparams.strides
        expand_ratio = self.hparams.expand_ratio

        # A first convolution on the original image to scale up the channel size
        if self.hparams.block_class == MobileNetBlock: # => Don't apply non-linearity on output
            self.input_net = nn.Sequential(
                nn.Conv2d(4, self.input_channel, kernel_size=3, padding=1, stride = 2, bias=False)
            )

        # Creating the MobileNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                output_channel = int(np.ceil(c_hidden[block_idx] * self.hparams.width_mult* 1./8) * 8) if bc > 1 else c_hidden[block_idx]
                if bc == 0:    
                    blocks.append(
                            self.hparams.block_class(c_in=self.input_channel,
                                                    act_fn=self.hparams.act_fn,
                                                    c_out=output_channel,
                                                    expand_ratio=expand_ratio[block_idx],
                                                    stride = strides[block_idx])
                        )
                else:
                    blocks.append(
                            self.hparams.block_class(c_in=self.input_channel,
                                                    act_fn=self.hparams.act_fn,
                                                    c_out=output_channel,
                                                    expand_ratio=expand_ratio[block_idx],
                                                    stride = 1)
                        )
                self.input_channel = output_channel

        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.Conv2d(c_hidden[-1], int(np.ceil(1280 * self.hparams.width_mult* 1./8) * 8), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(np.ceil(1280 * self.hparams.width_mult* 1./8) * 8)),
            self.hparams.act_fn(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(int(np.ceil(1280 * self.hparams.width_mult* 1./8) * 8), self.hparams.num_classes)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x
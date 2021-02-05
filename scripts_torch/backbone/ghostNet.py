import torch
from torch import nn
class GhostNet(nn.Module):
    def __init__(self, width_mult=1.0, out_stages=(4, 6, 9), act='ReLU', pretrain=True):
        super(GhostNet, self).__init__()
        self.width_mult = width_mult
        self.out_stages = out_stages
        # setting of inverted residual blocks
        self.cfgs = [
            # k, t,   c,  SE, s
            # stage1
            [[3, 16,  16, 0, 1]],     # 0
            # stage2
            [[3, 48,  24, 0, 2]],     # 1
            [[3, 72,  24, 0, 1]],     # 2  1/4
            # stage3
            [[5, 72,  40, 0.25, 2]],  # 3
            [[5, 120, 40, 0.25, 1]],  # 4  1/8
            # stage4
            [[3, 240, 80, 0, 2]],     # 5
            [[3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
             ],                       # 6  1/16
            # stage5
            [[5, 672, 160, 0.25, 2]], # 7
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]
             ]                        # 8
        ]
        #  ------conv+bn+act----------# 9  1/32

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = act_layers(act)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width_mult, 4)
                hidden_channel = _make_divisible(exp_size * width_mult, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    act=act, se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width_mult, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1, act=act)))  #9

        self.blocks = nn.Sequential(*stages)

        self._initialize_weights(pretrain)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        output = []
        for i in range(10):
            x = self.blocks[i](x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv_stem' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if pretrain:
            url = get_url(self.width_mult)
            if url is not None:
                state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
                self.load_state_dict(state_dict, strict=False)
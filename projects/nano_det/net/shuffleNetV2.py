from torch import nn
from net.shufflleV2 import ShuffleV2Block
from utils.activation import act_layers

class ShuffleNetV2(nn.Module):
    def __init__(self,
                 model_size='1.5x',
                 out_stages=(2, 3, 4),
                 with_last_conv=False,
                 kernal_size=3,
                 activation = 'ReLu'):
        super(ShuffleNetV2, self).__init__()
        print('shuffleNetV2 model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size

        #0:frist conv; 2,3,4:stacked conv block; 5:full connection layer
        if model_size == '0.5x':
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(

            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            act_layers(activation),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, self.stage_repeats, self._stage_out_channels[1:]):
            seq = [ShuffleV2Block(input_channels, output_channels, 2, activation=activation)]
            for i in range(repeats - 1):
                seq.append(ShuffleV2Block(output_channels, output_channels, 1, activation=activation))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            self.conv5 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                act_layers(activation),
            )
            self.stage4.add_module('conv5', self.conv5)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print('init shuffleNetV2 weights...')
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
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
        # if pretrain:
        #     url = model_urls['shufflenetv2_{}'.format(self.model_size)]
        #     if url is not None:
        #         pretrained_state_dict = model_zoo.load_url(url)
        #         print('=> loading pretrained model {}'.format(url))
        #         self.load_state_dict(pretrained_state_dict, strict=False)

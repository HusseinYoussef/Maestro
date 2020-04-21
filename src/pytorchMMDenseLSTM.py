import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
#from .utils import load_state_dict_from_url
from torch import Tensor
from torch.jit.annotations import List

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
        (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
        but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__( self, 
                growth_rate, 
                block_config_down,
                block_config_up,
                batch_zie=2,
                frame_rate=1000,
                num_channels=2,
                freq_bins=100,
                lstm_input=[],
                num_init_features=2,
                bn_size=4, 
                drop_rate=0,
                lstm_layers=1,
                num_classes=1000, memory_efficient=False,lstm_idx_down=-1,lstm_idx_up=-1):

        super(DenseNet, self).__init__()
        self.lstm_idx_down= lstm_idx_down
        self.lstm_idx_up= lstm_idx_up
        #initialization
        self.features_down = nn.Sequential()
        self.features_up = nn.Sequential()

        #Down sampling path
        num_features = num_init_features
        for i, num_layers in enumerate(block_config_down):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features_down.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config_down) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features_down.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        #lstm 
        self.lstmObjDown=nn.LSTM(input_size=lstm_input[0], hidden_size=lstm_input[0]//2,num_layers=lstm_layers ,bidirectional =True,batch_first=True )
        self.lstmObjUp=nn.LSTM(input_size=lstm_input[1], hidden_size=lstm_input[1]//2,num_layers=lstm_layers,bidirectional =True,batch_first=True )

        # Upsampling path
        for j, num_layers in enumerate(block_config_up,i+1):
            
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features * 2)
            self.features_up.add_module('transition%d' % (j + 1), trans)
            num_features = num_features * 2
            #print(num_features)
            
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features_up.add_module('denseblock%d' % (j + 1), block)
            num_features = num_features + num_layers * growth_rate
            

        # Final batch norm
        self.features_up.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i,module in enumerate(self.features_down):
            x=module(x)
            #print(x.shape)
            if i==self.lstm_idx_down:
                
                y=x.shape
                x= x.reshape(x.shape[0],x.shape[3],x.shape[1]*x.shape[2])
                x = self.lstmObjDown(x)[0]
                x=x.reshape(y[0],y[1],y[2],y[3])
            
        #lastLayer
        
        for i,module in enumerate(self.features_up):
            x=module(x)
            #print(x.shape)

            if i==self.lstm_idx_up:
                y=x.shape
                x= x.reshape(x.shape[0],x.shape[3],x.shape[1]*x.shape[2])
                x = self.lstmObjUp(x)[0]
                x=x.reshape(y[0],y[1],y[2],y[3])
        
        out = F.relu(x, inplace=True)
        #out = F.adaptive_avg_pool2d(out, (1, 1))
        #out = torch.flatten(out, 1)
        #out = self.classifier(out)
        return out


class MMDenseLSTM(nn.Module):
    def __init__( self):

        super(MMDenseLSTM, self).__init__()
        self.num_channels=2
        self.batchSize=2
        self.seq=600
        self.freqBins=240
        #the next lists values need to be changed when the training begins to fit the data
        self.lstmInputValuesBand1=[3930,5138]
        self.lstmInputValuesBand2=[900,1176]
        self.lstmInputValuesBand3=[240,25]# second number is dummy
        self.lstmInputValuesBandfull=[1710,1176]
        self.sampleInput =torch.rand(self.batchSize,self.num_channels,self.freqBins,self.seq)

        self.firstBandNet=DenseNet(
        growth_rate=14 ,
        block_config_down=(5, 5, 5, 5),
        block_config_up=(5,5,5),
        lstm_input=self.lstmInputValuesBand1,
        lstm_idx_down=6,
        lstm_idx_up=3,
        lstm_layers=1 #change to 128 when training begins

        )

        self.secBandNet=DenseNet(
            growth_rate=4 ,
            block_config_down=(4,4,4,4),
            block_config_up=(4,4,4),
            lstm_input=self.lstmInputValuesBand2,
            lstm_idx_down=6,
            lstm_idx_up=3,
            lstm_layers=1 #change to 32 when training begins
        )

        self.thirdBandNet=DenseNet(
            growth_rate=2,
            block_config_down=(1,1), # different from paper's arch
            block_config_up=(1,1),
            lstm_input=self.lstmInputValuesBand3,
            lstm_idx_down=4,
            lstm_idx_up=-1,
            lstm_layers=1 #change to 8 when training begins
        )

        self.fullBandNet=DenseNet(
            growth_rate=7 ,
            block_config_down=(3,3,4,5,5), # different from paper's arch
            block_config_up=(5,4,3,3),
            lstm_input=self.lstmInputValuesBandfull,
            lstm_idx_down=6,
            lstm_idx_up=-3,
            lstm_layers=1 #change to 128 when training begins
        )

    

    def forward(self):
        out1=self.firstBandNet(self.sampleInput)
        out2=self.secBandNet(self.sampleInput)
        out3=self.thirdBandNet(self.sampleInput)
        print(out1.shape)
        print(out2.shape)
        print(out3.shape)
        return out1

net =MMDenseLSTM()
net.forward()

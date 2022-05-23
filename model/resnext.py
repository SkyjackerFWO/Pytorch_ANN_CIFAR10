import re
from turtle import forward
import torch
import torch.nn as nn 
import torch.nn.functional as F

class block(nn.Module):
    expansion = 2
    def __init__(self, in_channels, cardinality = 32, bottleneck_width = 4, stride = 1):
        
        super(block,self).__init__()
        group_width = bottleneck_width * cardinality
        
        self.conv1 = nn.Conv2d(in_channels,group_width, kernel_size=1, bias=False )
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width,kernel_size=3, stride=stride,padding=1, groups=cardinality, bias=False) 
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, group_width*self.expansion, kernel_size=1, stride=1, bias = False)
        self.bn3 = nn.BatchNorm2d(group_width*self.expansion)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*group_width, kernel_size=1, stride=stride, bias= False),
                nn.BatchNorm2d(self.expansion*group_width)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality, bottleneck_width, image_channels,classes):
        """BUILD ResNeXt NETWORK

        Args:
            blocks (__): Block struct you wamt to build
            num_blocks (_arr_): number block for bulid net
            cardinality (_int_): cardinality
            bottleneck_width (_int_): bottleneck width
            image_channels (_int_): channels image (Red - Green - Blue: 3, Gray: 1)
            classes (_int_): class need to classification
        """
        
        super(ResNeXt,self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.layer1 = self._make_layer(block,num_blocks[0], 1)
        self.layer2 = self._make_layer(block,num_blocks[1], 2)
        self.layer3 = self._make_layer(block,num_blocks[2], 2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(cardinality*bottleneck_width*8, classes)
        
    def _make_layer(self, block ,num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides: 
            layers.append(block(in_channels=self.in_channels, cardinality= self.cardinality, bottleneck_width= self.bottleneck_width, stride=stride))
            self.in_channels = block.expansion*self.cardinality*self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage
        self.bottleneck_width*=2
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = F.avg_pool2d(x,8)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        
        return x
    
def RexNeXt29_2x64d():
    return ResNeXt(block,num_blocks=[3,3,3],cardinality=2,bottleneck_width=64, image_channels=3, classes=10)

def RexNeXt29_4x64d():
    return ResNeXt(block,num_blocks=[3,3,3],cardinality=4,bottleneck_width=64, image_channels=3, classes=10)

def RexNeXt29_8x64d():
    return ResNeXt(block,num_blocks=[3,3,3],cardinality=8,bottleneck_width=64, image_channels=3, classes=10)

def RexNeXt29_32x64d():
    return ResNeXt(block,num_blocks=[3,3,3],cardinality=32,bottleneck_width=64, image_channels=3, classes=10)

def test_resnext():
    net = RexNeXt29_2x64d()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())
    
# test_resnext()
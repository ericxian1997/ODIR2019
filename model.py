import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        #model_ft.avgpool = LocalAttentivePooling(4,1,True,'avg')
        self.model = model_ft

        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        
        #self.bns = nn.BatchNorm2d(2048)
        
        
        self.classifier = ClassBlock(4096, class_num)

    def forward(self, x1, x2):
        x1 = self.model.conv1(x1)
        x1 = self.model.bn1(x1)
        x1 = self.model.relu(x1)
        x1 = self.model.maxpool(x1)
        x1 = self.model.layer1(x1)
        x1 = self.model.layer2(x1)
        x1 = self.model.layer3(x1)
        x1 = self.model.layer4(x1)
        x1 = self.model.avgpool(x1)
        x1 = torch.squeeze(x1)

        x2 = self.model.conv1(x2)
        x2 = self.model.bn1(x2)
        x2 = self.model.relu(x2)
        x2 = self.model.maxpool(x2)
        x2 = self.model.layer1(x2)
        x2 = self.model.layer2(x2)
        x2 = self.model.layer3(x2)
        x2 = self.model.layer4(x2)
        x2 = self.model.avgpool(x2)
        x2 = torch.squeeze(x2)

        x = torch.cat((x1,x2),1)

        #x = x.view(x.size(0), 2048, -1)
        x = self.classifier(x)
        return x

        
# Define the ResNet50-based Model
class ft_net_152(nn.Module):

    def __init__(self, class_num):
        super(ft_net_152, self).__init__()
        model_ft = models.resnet152(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        #model_ft.avgpool = LocalAttentivePooling(4,1,True,'avg')
        self.model = model_ft

        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        
        #self.bns = nn.BatchNorm2d(2048)
        
        
        self.classifier = ClassBlock(2560, class_num)

    def forward(self, x1, x2):
        x1 = self.model.conv1(x1)
        x1 = self.model.bn1(x1)
        x1 = self.model.relu(x1)
        x1 = self.model.maxpool(x1)
        x1 = self.model.layer1(x1)
        x1 = self.model.layer2(x1)
        x1 = self.model.layer3(x1)
        x1 = self.model.layer4(x1)
        x1 = self.model.avgpool(x1)
        x1 = torch.squeeze(x1)

        x2 = self.model.conv1(x2)
        x2 = self.model.bn1(x2)
        x2 = self.model.relu(x2)
        x2 = self.model.maxpool(x2)
        x2 = self.model.layer1(x2)
        x2 = self.model.layer2(x2)
        x2 = self.model.layer3(x2)
        x2 = self.model.layer4(x2)
        x2 = self.model.avgpool(x2)
        x2 = torch.squeeze(x2)

        x = torch.cat((x1,x2),1)

        #x = x.view(x.size(0), 2048, -1)
        x = self.classifier(x)
        return x


class ft_efnet(nn.Module):

    def __init__(self, class_num):
        super(ft_efnet, self).__init__()
        model_ft = EfficientNet.from_pretrained('efficientnet-b3')
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))

        #model_ft.avgpool = LocalAttentivePooling(4,1,True,'avg')
        self.model = model_ft

        # remove the final downsample
        #self.model.layer4[0].downsample[0].stride = (1,1)
        #self.model.layer4[0].conv2.stride = (1,1)
        
        #self.bns = nn.BatchNorm2d(2048)
        
        
        self.classifier = ClassBlock(3072, class_num)

    def forward(self, x1, x2):
        x1 = self.model.extract_features(x1)
        x1 = self.model.avgpool(x1)
        x1 = torch.squeeze(x1)

        x2 = self.model.extract_features(x2)
        x2 = self.model.avgpool(x2)
        x2 = torch.squeeze(x2)

        x = torch.cat((x1,x2),1)

        #x = x.view(x.size(0), 2048, -1)
        x = self.classifier(x)
        return x
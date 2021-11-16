import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0,0,0,0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        
        return out3




class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(64, 64 // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(64 // 16, 64, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class Attention(nn.Module):
    def __init__(self):
        super(Attention , self).__init__()
        
        self.att_c2 = ChannelAttention()
        self.att_c3 = ChannelAttention()
        self.att_c4 = ChannelAttention()
        self.att_c5 = ChannelAttention()

        self.att_s2 = SpatialAttention()
        self.att_s3 = SpatialAttention()
        self.att_s4 = SpatialAttention()
        self.att_s5 = SpatialAttention()

    def forward(self, x2,x3,x4,x5):

        tempt2 = x2.mul(self.att_c2(x2))
        tempt2 = tempt2.mul(self.att_s2(tempt2))

        tempt3 = x3.mul(self.att_c3(x3))
        tempt3 = tempt3.mul(self.att_s3(tempt3))

        tempt4 = x4.mul(self.att_c4(x4))
        tempt4 = tempt4.mul(self.att_s4(tempt4))

        tempt5 = x5.mul(self.att_c5(x5))
        tempt5 = tempt5.mul(self.att_s5(tempt5))
        
        return tempt2, tempt3, tempt4, tempt5



class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




class LightRFB(nn.Module):
    def __init__(self, c_in=128, c_out=64):
        super(LightRFB, self).__init__()
        self.br2 = nn.Sequential(
            BasicConv(c_in, c_out, kernel_size=1,bias=False, bn=True, relu=True),

            BasicConv(c_out, c_out, kernel_size=3, dilation=1, padding=1, groups=c_out, bias=False,
                      relu=False),
        )
        self.br3 = nn.Sequential(
            BasicConv(c_out, c_out, kernel_size=3, dilation=1, padding=1, groups=c_out, bias=False,
                      bn=True,relu=False),
            BasicConv(c_out, c_out, kernel_size=1, dilation=1, bias=False,bn=True,relu=True),

            BasicConv(c_out, c_out, kernel_size=3, dilation=3, padding=3, groups=c_out, bias=False,
                      relu=False),
        )
        self.br4 = nn.Sequential(
            BasicConv(c_out, c_out, kernel_size=5, dilation=1, padding=2, groups=c_out, bias=False,
                      bn=True, relu=False),
            BasicConv(c_out, c_out, kernel_size=1, dilation=1, bias=False, bn=True, relu=True),

            BasicConv(c_out, c_out, kernel_size=3, dilation=5, padding=5, groups=c_out, bias=False,
                      relu=False),
        )
        self.br5 = nn.Sequential(
            BasicConv(c_out, c_out, kernel_size=7, dilation=1, padding=3, groups=c_out, bias=False,
                      bn=True, relu=False),
            BasicConv(c_out, c_out, kernel_size=1, dilation=1, bias=False, bn=True, relu=True),

            BasicConv(c_out, c_out, kernel_size=3, dilation=7, padding=7, groups=c_out, bias=False,
                      relu=False),
        )


        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b   = nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b   = nn.BatchNorm2d(64)

        self.conv1d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d   = nn.BatchNorm2d(64)
        self.conv2d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d   = nn.BatchNorm2d(64)
        self.conv3d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d   = nn.BatchNorm2d(64)
        self.conv4d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d   = nn.BatchNorm2d(64)


    def forward(self, x):
        
        out2=self.br2(x)

        x3=self.br3(out2)
        out3 = F.max_pool2d(x3, kernel_size=2, stride=2)

        x4=self.br4(out3)
        out4 = F.max_pool2d(x4, kernel_size=2, stride=2)

        x5=self.br5(out4)
        out5 = F.max_pool2d(x5, kernel_size=2, stride=2)


        out1b = F.relu(self.bn1b(self.conv1b(out2)), inplace=True)
        out2b = F.relu(self.bn2b(self.conv2b(out3)), inplace=True)
        out3b = F.relu(self.bn3b(self.conv3b(out4)), inplace=True)
        out4b = F.relu(self.bn4b(self.conv4b(out5)), inplace=True)

        out1d = F.relu(self.bn1d(self.conv1d(out2)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out3)), inplace=True)
        out3d = F.relu(self.bn3d(self.conv3d(out4)), inplace=True)
        out4d = F.relu(self.bn4d(self.conv4d(out5)), inplace=True)
        
        return (out4b, out3b, out2b, out1b), (out4d, out3d, out2d, out1d)



    
#MCNet
class RGBTNet(nn.Module):
    def __init__(self):
        super(RGBTNet, self).__init__()
        
        #Backbone model
        self.res_image = ResNet50()
        self.res_image.load_state_dict(models.resnet50(pretrained=True).state_dict(), strict=False)
        self.res_thermal   = ResNet50()
        self.res_thermal.load_state_dict(models.resnet50(pretrained=True).state_dict(), strict=False)

        self.bi5   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.bi4   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.bi3   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.bi2   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.bt5   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.bt4   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.bt3   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.bt2   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.att_t = Attention()
        self.att_i = Attention()

        self.rfb = LightRFB()
        self.decoderi = Decoder()
        self.decodert = Decoder()
        self.lineari  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineart  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        

    def forward(self, image, thermal, shape=None):
        i1,i2,i3,i4,i5 = self.res_image(image)
        t1,t2,t3,t4,t5 = self.res_thermal(thermal)

        i2, i3, i4, i5 = self.bi2(i2), self.bi3(i3), self.bi4(i4), self.bi5(i5)
        t2, t3, t4, t5 = self.bt2(t2), self.bt3(t3), self.bt4(t4), self.bt5(t5)

        att2i, att3i, att4i, att5i = self.att_i(i2, i3, i4, i5)
        att2t, att3t, att4t, att5t = self.att_t(t2, t3, t4, t5)

        out2i = i2 + att2t
        out3i = i3 + att3t
        out4i = i4 + att4t
        out5i = i5 + att5t

        out2t = t2 + att2i
        out3t = t3 + att3i
        out4t = t4 + att4i
        out5t = t5 + att5i


        outi1 = self.decoderi([out5i, out4i, out3i, out2i])
        outt1 = self.decodert([out5t, out4t, out3t, out2t])

        out1  = torch.cat([outi1, outt1], dim=1)

        outi2, outt2 = self.rfb(out1)

        outi2 = self.decoderi([out5i, out4i, out3i, out2i], outi2)
        outt2 = self.decodert([out5t, out4t, out3t, out2t], outt2)

        out2  = torch.cat([outi2, outt2], dim=1)

        if shape is None:
            shape = image.size()[2:]
        out1  = F.interpolate(self.linear(out1),   size=shape, mode='bilinear')
        outi1 = F.interpolate(self.lineari(outi1), size=shape, mode='bilinear')
        outt1 = F.interpolate(self.lineart(outt1), size=shape, mode='bilinear')

        out2  = F.interpolate(self.linear(out2),   size=shape, mode='bilinear')
        outi2 = F.interpolate(self.lineari(outi2), size=shape, mode='bilinear')
        outt2 = F.interpolate(self.lineart(outt2), size=shape, mode='bilinear')
        
        return outi1, outt1, out1, outi2, outt2, out2
    
    
    
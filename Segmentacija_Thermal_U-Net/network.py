import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(UNet, self).__init__()

        # Encoder
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1) # konvolucija, jedro 3x3, 1 piksel za robove
        self.relu1_1 = nn.ReLU(inplace=True) # ReLu nelinearnost
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # manjšanje dimenzije, 2x2 jedro, manjša dimenzijo za 2

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (latentni prostor)
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bottleneck_relu2 = nn.ReLU(inplace=True)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # Obratna konvolucija (parametri so isti)
        self.conv4_3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_4 = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_4 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.conv2_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_4 = nn.ReLU(inplace=True)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_4 = nn.ReLU(inplace=True)

        # Final layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1) # 1 out za masko

    # Propagacija skozi mrežo
    def forward(self, x):
        
        # Encoder
        e1 = self.relu1_1(self.conv1_1(x))
        e1 = self.relu1_2(self.conv1_2(e1))
        p1 = self.pool1(e1)

        e2 = self.relu2_1(self.conv2_1(p1))
        e2 = self.relu2_2(self.conv2_2(e2))
        p2 = self.pool2(e2)

        e3 = self.relu3_1(self.conv3_1(p2))
        e3 = self.relu3_2(self.conv3_2(e3))
        p3 = self.pool3(e3)

        e4 = self.relu4_1(self.conv4_1(p3))
        e4 = self.relu4_2(self.conv4_2(e4))
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck_relu1(self.bottleneck_conv1(p4))
        b = self.bottleneck_relu2(self.bottleneck_conv2(b))

        # Decoder
        d4 = self.upconv4(b)
        #d4 = torch.cat((d4, e4), dim=1) # direktna povezava (siva puščica)
        diffY = e4.size()[2] - d4.size()[2]
        diffX = e4.size()[3] - d4.size()[3]
        d4 = F.pad(d4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.relu4_3(self.conv4_3(d4))
        d4 = self.relu4_4(self.conv4_4(d4))

        d3 = self.upconv3(d4)
        #d3 = torch.cat((d3, e3), dim=1) # direktna povezava (siva puščica)
        diffY = e3.size()[2] - d3.size()[2]
        diffX = e3.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.relu3_3(self.conv3_3(d3))
        d3 = self.relu3_4(self.conv3_4(d3))

        d2 = self.upconv2(d3)
        #d2 = torch.cat((d2, e2), dim=1) # direktna povezava (siva puščica)
        diffY = e2.size()[2] - d2.size()[2]
        diffX = e2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.relu2_3(self.conv2_3(d2))
        d2 = self.relu2_4(self.conv2_4(d2))

        d1 = self.upconv1(d2)
        #d1 = torch.cat((d1, e1), dim=1) # direktna povezava (siva puščica)
        diffY = e1.size()[2] - d1.size()[2]
        diffX = e1.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.relu1_3(self.conv1_3(d1))
        d1 = self.relu1_4(self.conv1_4(d1))

        # Final layer
        out = self.final_conv(d1)
        return out


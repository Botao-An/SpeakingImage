import torch.nn as nn
import torch

class parallel_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(alpha=1.0, inplace=True)
        )
    def forward(self, input):
        return self.layer(input)

class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(alpha=1.0, inplace=True)
        )
    def forward(self, input):
        return self.layer(input)

class decoder_block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1),
            nn.BatchNorm2d(middle_channels),
            nn.ELU(alpha=1.0, inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, input):

        return self.layer(input)


class DaZhuangNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.h_parallel_layer1 = parallel_block(4, 32)
        self.h_down_layer1 = encoder_block(32, 64)
        self.h_down_layer2 = encoder_block(64, 128)
        self.h_down_layer3 = encoder_block(128, 256)
        self.h_down_layer4 = encoder_block(256, 512)
        self.h_down_layer5 = encoder_block(512, 1024)
        self.h_center_layer1 = parallel_block(1024, 2048)
        self.h_center_layer2 = parallel_block(2048, 1024)
        self.h_up_layer5 = decoder_block(2048, 1024, 512)
        self.h_up_layer4 = decoder_block(1024, 512, 256)
        self.h_up_layer3 = decoder_block(512, 256, 128)
        self.h_up_layer2 = decoder_block(256, 128, 64)
        self.h_up_layer1 = decoder_block(128, 64, 32)
        self.h_parallel_layer2 = parallel_block(64, 32)
        self.h_final = nn.Conv2d(32, 3, kernel_size=1)

        self.e_parallel_layer1 = parallel_block(3, 32)
        self.e_down_layer1 = encoder_block(32, 64)
        self.e_down_layer2 = encoder_block(64, 128)
        self.e_down_layer3 = encoder_block(128, 256)
        self.e_down_layer4 = encoder_block(256, 512)
        self.e_center_layer1 = parallel_block(512, 1024)
        self.e_center_layer2 = parallel_block(1024, 512)
        self.e_up_layer4 = decoder_block(1024, 512, 256)
        self.e_up_layer3 = decoder_block(512, 256, 128)
        self.e_up_layer2 = decoder_block(256, 128, 64)
        self.e_up_layer1 = decoder_block(128, 64, 32)
        self.e_parallel_layer2 = parallel_block(64, 32)
        self.e_final = nn.Conv2d(32, 1, kernel_size=1)


    def forward(self, cover, secret):

        mix_input = torch.cat((cover, secret), 1)
        h_ec1 = self.h_parallel_layer1(mix_input)
        h_ec2 = self.h_down_layer1(h_ec1)
        h_ec3 = self.h_down_layer2(h_ec2)
        h_ec4 = self.h_down_layer3(h_ec3)
        h_ec5 = self.h_down_layer4(h_ec4)
        h_ec6 = self.h_down_layer5(h_ec5)
        h_center = self.h_center_layer1(h_ec6)
        h_dc6 = self.h_center_layer2(h_center)
        h_dc5 = self.h_up_layer5(torch.cat((h_dc6,h_ec6), 1))
        h_dc4 = self.h_up_layer4(torch.cat((h_dc5,h_ec5), 1))
        h_dc3 = self.h_up_layer3(torch.cat((h_dc4,h_ec4), 1))
        h_dc2 = self.h_up_layer2(torch.cat((h_dc3,h_ec3), 1))
        h_dc1 = self.h_up_layer1(torch.cat((h_dc2, h_ec2), 1))
        h_pro = self.h_parallel_layer2(torch.cat((h_dc1, h_ec1), 1))
        stego_image = self.h_final(h_pro)

        e_ec1 = self.e_parallel_layer1(stego_image)
        e_ec2 = self.e_down_layer1(e_ec1)
        e_ec3 = self.e_down_layer2(e_ec2)
        e_ec4 = self.e_down_layer3(e_ec3)
        e_ec5 = self.e_down_layer4(e_ec4)
        e_center = self.e_center_layer1(e_ec5)
        e_dc5 = self.e_center_layer2(e_center)
        e_dc4 = self.e_up_layer4(torch.cat((e_dc5, e_ec5), 1))
        e_dc3 = self.e_up_layer3(torch.cat((e_dc4, e_ec4), 1))
        e_dc2 = self.e_up_layer2(torch.cat((e_dc3, e_ec3), 1))
        e_dc1 = self.e_up_layer1(torch.cat((e_dc2, e_ec2), 1))
        e_pro = self.e_parallel_layer2(torch.cat((e_dc1, e_ec1), 1))
        out_secret = self.e_final(e_pro)

        return stego_image, out_secret

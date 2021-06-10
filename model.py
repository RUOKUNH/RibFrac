import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k_size = 3, stride = 1, padding = 1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size = k_size, stride = stride, padding = padding)
        self.batch_norm = nn.BatchNorm3d(out_channel)
        self.relu = self.relu = nn.LeakyReLU(inplace = True)
        
    def forward(self, x):
        x = self.conv3d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channel, model_depth = 4, pool_size = 2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channel = 2 ** (depth+1) *self.root_feat_maps
            for i in range(self.num_conv_blocks):
                self.conv_block = ConvBlock(in_channel, feat_map_channel)
                self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                in_channel, feat_map_channel = feat_map_channel, 2*feat_map_channel
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size = pool_size, stride = 2, padding = 0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling
        
    def forward(self, x):
        sample_features = {}
        for k, layer in self.module_dict.items():
            x = layer(x)
            # if k.startswith('conv'):
            #     print('conv')
            # else:
            #     print('pooling')
            # print(x.shape)
            if k.startswith("conv") and int(k[-1]) == self.num_conv_blocks-1:
                sample_features[int(k.split('_')[1])] = x
        return x, sample_features
    
class ConvTranspose(nn.Module):
    def __init__(self, in_channel, out_channel, k_size = 3, stride = 2, padding = 1, output_padding = 1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channel, out_channel, k_size, stride, padding, output_padding)
        self.batch_norm = nn.BatchNorm3d(num_features = out_channel)
        self.relu = nn.LeakyReLU(inplace = True)
        
    def forward(self, x):
        x = self.conv3d_transpose(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, out_channel, model_depth = 4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 16
        self.module_dict = nn.ModuleDict()
        
        for depth in range(model_depth-2, -1, -1):
            feat_map_channel = 2 ** (depth+1) * self.num_feat_maps
            self.deconv = ConvTranspose(feat_map_channel * 4, feat_map_channel * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(feat_map_channel * 6, feat_map_channel * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(feat_map_channel * 2, feat_map_channel * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            
            if depth == 0:
                self.final_conv = ConvBlock(feat_map_channel * 2, out_channel)
                self.module_dict["final_conv"] = self.final_conv
        
    def forward(self, x, down_sampling_features):
         for k,layer in self.module_dict.items():
             if k.startswith("deconv"):
                 # print("deconv")
                 x = layer(x)
                 # print(x.shape)
                 # print("cat")
                 x = torch.cat((down_sampling_features[int(k[-1])], x), dim = 1)
                 # print(x.shape)
             else:
                 # print("conv")
                 x = layer(x)
                 # print(x.shape)
         return x
     
class UNet(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 1):
        super(UNet, self).__init__()
        # self.encoder = EncoderBlock(in_channel)
        # self.decoder = DecoderBlock(out_channel)
        self.pooling = nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 0)
        self.encoder1 = nn.Sequential(
            ConvBlock(in_channel, 8),
            ConvBlock(8, 16),
            )
        self.encoder2 = nn.Sequential(
            ConvBlock(16, 16),
            ConvBlock(16, 32),
            )
        self.encoder3 = nn.Sequential(
            ConvBlock(32, 32),
            ConvBlock(32, 64),
            )
        self.encoder4 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 128),
            )
        
        self.deconv1 = ConvTranspose(128, 128)
        self.deconv2 = ConvTranspose(64, 64)
        self.deconv3 = ConvTranspose(32, 32)
        
        self.decoder1 = nn.Sequential(
            ConvBlock(192, 64),
            ConvBlock(64, 64),
            )
        self.decoder2 = nn.Sequential(
            ConvBlock(96, 32),
            ConvBlock(32, 32)
            )
        self.decoder3 = nn.Sequential(
            ConvBlock(48, 16),
            ConvBlock(16, 16),
            )
        
        # self.final_conv = ConvBlock(16, 1)
        self.final_conv = nn.Conv3d(16, 1, 3, 1, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pooling(encoder1))
        encoder3 = self.encoder3(self.pooling(encoder2))
        encoder4 = self.encoder4(self.pooling(encoder3))
        
        decoder1 = torch.cat((encoder3, self.deconv1(encoder4)), dim = 1)
        decoder1 = self.decoder1(decoder1)
        decoder2 = torch.cat((encoder2, self.deconv2(decoder1)), dim = 1)
        decoder2 = self.decoder2(decoder2)
        decoder3 = torch.cat((encoder1, self.deconv3(decoder2)), dim = 1)
        decoder3 = self.decoder3(decoder3)
        
        out = self.final_conv(decoder3)
        
        return out
    
    
class UNet2(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 1):
        super(UNet2, self).__init__()
        # self.encoder = EncoderBlock(in_channel)
        # self.decoder = DecoderBlock(out_channel)
        self.pooling = nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 0)
        self.encoder1 = nn.Sequential(
            ConvBlock(in_channel, 32),
            ConvBlock(32, 64),
            )
        self.encoder2 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 128),
            )
        self.encoder3 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 256),
            )
        self.encoder4 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 512),
            )
        
        self.deconv1 = ConvTranspose(512, 512)
        self.deconv2 = ConvTranspose(256, 256)
        self.deconv3 = ConvTranspose(128, 128)
        
        self.decoder1 = nn.Sequential(
            ConvBlock(768, 256),
            ConvBlock(256, 256),
            )
        self.decoder2 = nn.Sequential(
            ConvBlock(384, 128),
            ConvBlock(128, 128)
            )
        self.decoder3 = nn.Sequential(
            ConvBlock(192, 64),
            ConvBlock(64, 64),
            )
        
        # self.final_conv = ConvBlock(16, 1)
        self.final_conv = nn.Conv3d(64, 1, 3, 1, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pooling(encoder1))
        encoder3 = self.encoder3(self.pooling(encoder2))
        encoder4 = self.encoder4(self.pooling(encoder3))
        
        decoder1 = torch.cat((encoder3, self.deconv1(encoder4)), dim = 1)
        decoder1 = self.decoder1(decoder1)
        decoder2 = torch.cat((encoder2, self.deconv2(decoder1)), dim = 1)
        decoder2 = self.decoder2(decoder2)
        decoder3 = torch.cat((encoder1, self.deconv3(decoder2)), dim = 1)
        decoder3 = self.decoder3(decoder3)
        
        out = self.final_conv(decoder3)
        
        return out
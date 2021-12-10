import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SpectralNorm
import numpy as np
from torch.nn import init
import functools

def defineG(opt, norm="batch", init_type="normal"):
    """
    norm: batch | instance
    """
    norm_layer = get_norm_layer(norm)
    # if netsize == 64:
    #     net = UnetSkip_RGB64(opt, norm_layer)
    # elif netsize ==  128:
    #     net = UnetSkip_RGB128(opt, norm_layer)
    # elif netsize == 256:
    #     net = UnetSkip_RGB256(opt, norm_layer)
    net = GenNetwork(img_resolution=opt.img_size, 
                                img_chans=opt.in_nc,
                                lat_dim=opt.lat_dim, 
                                channel_max=512,
                                norm_layer=norm_layer)
    init_weights(net, init_type)

    return net

def defineD(opt, norm="batch", init_type="normal"):
    norm_layer = get_norm_layer(norm)
    net = Discriminator(img_resolution=opt.img_size, 
                            img_chans=opt.in_nc,
                            channel_max=512,
                            norm_layer=norm_layer)
    init_weights(net)

    return net

def defineDL(opt, init_type="normal"):
    net = Dis_lat(opt)
    init_weights(net, init_type)
    return net
#----------------------------define DL---------------------------------------------------------#
class Dis_lat(nn.Module):
    def __init__(self, opt):
        super(Dis_lat, self).__init__()
        self.opt = opt
        self.dl_dim = self.opt.lat_dim

        self.linear = nn.Sequential(
            nn.Linear(self.dl_dim, self.dl_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.dl_dim, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        output = self.linear(inputs)
        return output

#----------------------------define DL---------------------------------------------------------#
class enBlock(nn.Module):
    """
    encoder contain:
        1.4X4, striede =2, padding=1
        2.bn
        3.LeakyReLU(0.2, inplace=True)
    """
    def __init__(self, input_nc, output_nc, normer):
        super(enBlock, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.input_nc, self.output_nc,
                     kernel_size=4, stride=2, padding=1, bias=False)),
            normer(self.output_nc),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class  DeBlock(nn.Module):
    def __init__(self, input_nc, output_nc, normer=nn.BatchNorm2d):
        super(DeBlock, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.conv = SpectralNorm(nn.ConvTranspose2d(self.input_nc, self.output_nc,
                                       kernel_size=4, stride=2, padding=1, bias=False))
        self.bn = normer(self.output_nc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_pred, inputs):
        x = torch.cat([x_pred, inputs], dim=1) # 在通道上进行拼接
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#--------------------------------------new core coding, save memory begin----------------------------------------------#
class EncoderMidBlock(nn.Module):
    def __init__(self,
                            in_chans, 
                            lat_dim,
                            norm_layer=nn.BatchNorm2d,
                            leaky_rate=0.2,
                            isInplace=True):
        super().__init__()
        self.in_channel = in_chans
        self.lat_dim = lat_dim
        self.leaky_rate = leaky_rate
        self.isInplace = isInplace
        self.norm = norm_layer if norm_layer else nn.Identity()

        self.conv0 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.in_channel, self.lat_dim, kernel_size=4, stride=1, bias=False)),
            self.norm(self.lat_dim),
            nn.LeakyReLU(negative_slope=self.leaky_rate, inplace=self.isInplace)
        )

        self.conv1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.lat_dim, self.in_channel, kernel_size=4, stride=1, bias=False)),
            self.norm(self.in_channel),
            nn.ReLU(inplace=self.isInplace)
            )

    def forward(self, x):
        x = self.conv0(x)
        x_o = x
        x = self.conv1(x)
        return x, x_o

class EncoderBlock(nn.Module):
    def __init__(self, in_channel, 
                                        tmp_channel, 
                                        out_channel, 
                                        img_channel,
                                        img_resolution,
                                        leaky_rate = 0.2, 
                                        isInplace = True,
                                        norm_layer=nn.BatchNorm2d):
        super(EncoderBlock, self).__init__()
        self.in_channel = in_channel
        self.tmp_channel = tmp_channel
        self.out_channel = out_channel
        self.img_channel = img_channel
        self.img_resolution = img_resolution
        self.norm = norm_layer if norm_layer else nn.Identity()
        
        self.leaky_rate = leaky_rate
        self.isInplace = isInplace

        if in_channel == 0:
            self.fromrgb = nn.Sequential(
                SpectralNorm(nn.Conv2d(self.img_channel, self.tmp_channel, kernel_size=1, stride=1, bias=True)),
                norm_layer(self.tmp_channel),
                nn.LeakyReLU(negative_slope=self.leaky_rate,inplace=self.isInplace)
                )
        
        self.conv0 = nn.Sequential(
                SpectralNorm(nn.Conv2d(self.tmp_channel, self.tmp_channel, kernel_size=3, stride=1,padding=1, bias=True)),
                norm_layer(self.tmp_channel),
                nn.LeakyReLU(negative_slope=self.leaky_rate,inplace=self.isInplace)
                )

        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.tmp_channel, self.out_channel, kernel_size=4, stride=2, padding=1,bias=True)),
            norm_layer(self.out_channel),
            nn.LeakyReLU(negative_slope=self.leaky_rate,inplace=self.isInplace)
            )
    def forward(self, x):
        if self.in_channel == 0:
            x = self.fromrgb(x)
        
        x = self.conv0(x)
        x = self.conv1(x)
        return x


class EncoderUPBlock(nn.Module):
    def __init__(self,
                            in_chans, 
                            out_chans, 
                            isPlace= True,
                            norm_layer=nn.BatchNorm2d):
        super(EncoderUPBlock,self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.isPlace = isPlace
        self.norm = norm_layer if norm_layer else nn.Identity()

        self.conv = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(in_chans, out_chans, kernel_size=4, stride=2,padding=1, bias=False)),
            self.norm(self.out_chans),
            nn.ReLU(inplace=self.isPlace)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class GenNetwork(nn.Module):
    def __init__(self, 
                                img_resolution, 
                                img_chans,
                                lat_dim, 
                                channel_max=512,
                                norm_layer=nn.BatchNorm2d):
        super(GenNetwork, self).__init__()
        channel_base_dict = {32:1024, 64:2048, 128:4096, 256:8192, 512:16384, 1024:32769}
        channel_base = channel_base_dict[img_resolution]

        self.img_resolution = img_resolution
        self.img_chans = img_chans
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_resolution_block = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.norm = norm_layer if norm_layer else nn.Identity()
        self.lat_dim = lat_dim

        channel_dict = {res:min(channel_base // res, channel_max ) for res in self.img_resolution_block + [4]}

        # build encoder network
        for idx, res in enumerate(self.img_resolution_block):
            in_channel = channel_dict[res] if res < img_resolution else 0
            tmp_channel = channel_dict[res] 
            out_channel = channel_dict[res // 2]

            block = EncoderBlock(in_channel, tmp_channel, out_channel, img_chans, img_resolution, norm_layer=self.norm)
            setattr(self, f"gen_{res}", block)
        
        # mid layer
        self.midBlock = EncoderMidBlock(channel_dict[4], self.lat_dim, self.norm)

        #  build decoder network
        self.img_resolution_block = self.img_resolution_block + [4] #[256, 128, 64, 32, 8, 4]
        for idx, res in enumerate(self.img_resolution_block[::-1][:-1]):
            in_channel = channel_dict[res] * 2 
            out_channel = channel_dict[res * 2] if res < self.img_resolution // 2 else channel_dict[res]

            disBlock = EncoderUPBlock(in_channel, out_channel, norm_layer=self.norm)
            setattr(self, f"dis_{res}", disBlock)
        
        self.last_conv = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(channel_dict[img_resolution // 2] * 2, img_chans, kernel_size=4, stride=2,padding=1, bias=False)),
                                                                    nn.Tanh())
    
    def forward_feature(self, x):
        mid_feature = list()
        for idx, res in enumerate(self.img_resolution_block[:-1]):
            block = getattr(self, f"gen_{res}")
            x = block(x)
            mid_feature.append(x)
            print(f"encode {res}, x.shape={x.shape}")
        
        # middle layer
        x, x_o = self.midBlock(x)

        mid_feature = mid_feature[::-1]
        # upsample layer
        for idx, res in enumerate(self.img_resolution_block[::-1][:-2]):
            block = getattr(self, f"dis_{res}")
            # print(f"res = {res}, x.shape= {x.shape}, mid.shape = {mid_feature[idx].shape}")
            x = torch.cat((x, mid_feature[idx]), dim=1)
            x = block(x)

        # print(f"final : x.shape= {x.shape}, mid.shape = {mid_feature[-1].shape}")
        # fianl last layer
        x = torch.cat((x, mid_feature[-1]), dim=1)
        return x, x_o

    def forward(self, x):
        x, x_mid = self.forward_feature(x)
        x = self.last_conv(x)
        return x, x_mid

#--------------------------------------new core coding, save memory end----------------------------------------------#

class Discriminator(nn.Module):
    def __init__(self,
                            img_resolution, 
                            img_chans,
                            channel_max=512,
                            norm_layer=nn.BatchNorm2d):
        super().__init__()
        channel_base_dict = {32:1024, 64:2048, 128:4096, 256:8192, 512:16384, 1024:32769}
        chanel_base = channel_base_dict[img_resolution]
        self.img_channel = img_chans
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(self.img_resolution))
        self.img_resolution_block = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.norm = norm_layer if norm_layer else nn.Identity()

        channel_dict = {res:min(chanel_base // res, channel_max) for res in self.img_resolution_block + [4]}
        
        for idx, res in enumerate(self.img_resolution_block):
            in_channel = channel_dict[res] if res < self.img_resolution else 0
            tmp_channel = channel_dict[res]
            out_channel = channel_dict[res // 2]
            block = EncoderBlock(in_channel, tmp_channel, out_channel, img_chans, img_resolution, norm_layer=self.norm)

            setattr(self, f"{res}", block)
        
        self.last_conv = SpectralNorm(nn.Conv2d(channel_dict[4], channel_dict[4], kernel_size=4, stride=1,bias=False))
        self.linear = nn.Linear(channel_dict[4], 1, bias=True)
    
    def forward(self, x):
        
        for idx, res in enumerate(self.img_resolution_block):
            block = getattr(self, f"{res}")
            x = block(x)
        
        x_mid = x
        x = torch.squeeze(self.last_conv(x))
        x = self.linear(x)
        return x, x_mid
#-------------------------------------公共部分开始----------------------------#
# 正则化
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# 初始化权重
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
#-------------------------------------公共部分结束----------------------------#

"""
class Discriminator(nn.Module):
    def __init__(self, opt, norm):
    # def __init__(self, nc=3, isize=64,d_dim=64, norm=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        self.opt = opt
        self.img_nc = self.opt.nc
        self.d_dim = self.opt.d_dim
        
        isize = self.opt.isize

        # self.img_nc = nc
        # self.d_dim = d_dim
        
        # self.isize = isize
        self.model = nn.Sequential()

        self.model.add_module("conv-{0}-{1}".format(self.img_nc, self.d_dim),SpectralNorm(nn.Conv2d(self.img_nc, self.d_dim, 4, 2, 1, bias=False)))
        self.model.add_module("norm-{0}".format(self.d_dim),norm(self.d_dim))
        self.model.add_module("inittial-realu-{0}".format(self.d_dim),nn.LeakyReLU(0.2, inplace=True))
        
        isize = isize // 2

        while(isize > 4):
            self.model.add_module("initial-conv-{0}-{1}-conv".format(self.d_dim, self.d_dim *2),
                                                            SpectralNorm(nn.Conv2d(self.d_dim, self.d_dim * 2, 4, 2, 1, bias=False)))
            self.model.add_module("initial-batchnorm-{0}".format(self.d_dim * 2), norm(self.d_dim * 2))
            self.model.add_module("initial-{0}-relu".format(self.d_dim * 2), nn.LeakyReLU(0.2, inplace=True))

            isize = isize // 2
            self.d_dim *= 2
            
   
        # 这里考虑增加个特征图损失
        # self.model.add_module("final-conv-{0}-{1}".format(self.d_dim , self.d_dim ),SpectralNorm(nn.Conv2d(self.d_dim, self.d_dim, kernel_size=4,stride=1, padding=0,bias=False)))
        self.last_conv = SpectralNorm(nn.Conv2d(self.d_dim, self.d_dim, kernel_size=4,stride=1, padding=0,bias=False))


        self.line = nn.Linear(self.d_dim, 1, bias=False)
        
        # 这部分是因为不用wgan-dp
        self.final_act = nn.Sigmoid()
    
    def forward(self,input):
        n, h, w, c = input.size()
        x = self.model(input)
        feature = x
        x = self.last_conv(x)
        x = x.view(n, -1)
        x = self.line(x)
        
        x = self.final_act(x) #如果要用wgan-dp,这行代码需要加注释
        # x = x.view(-1, 1).squeeze(1)

        return x, feature
"""


class UnetSkip_RGB64(nn.Module):
    def __init__(self, opt, normer=nn.BatchNorm2d):
    # def __init__(self, img_nc=3, g_dim=64, lat_dim=100, normer=nn.BatchNorm2d):
        """
        第一层先经过一个3*3卷积，步长为1;再经过一个4×4，步长为2
        :param img_nc: image input channel:3
        :param g_nc: final output, innermost latent
        :param norm:批量归一化方式.BatchNorm2d | InStance
        """
        super(UnetSkip_RGB64, self).__init__()
        self.opt = opt
        self.img_nc = self.opt.nc
        self.g_nc = self.opt.g_dim
        self.lat_dim = self.opt.lat_dim

        # (N, 3, h, w) -> (N, 64, h/2, w/2)
        self.down1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.img_nc, self.g_nc,
                                   kernel_size=3, stride=1, padding=1, bias=False)),
            normer(self.g_nc),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(self.g_nc, self.g_nc * 2,
                                   kernel_size=4, stride=2, padding=1, bias=False)),
            normer(self.g_nc * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )  #  h / 2

        self.down1_out = self.down1

        # (N, 64, h, w) -> (N, 128, h / 2, w /  2)
        self.down2 = enBlock(self.g_nc * 2, self.g_nc * 4, normer)  # h / 4
        # (N, 128, h / 2, w / 2) -> (N, 256, h / 4, w / 4)
        self.down3 = enBlock(self.g_nc * 4, self.g_nc * 8, normer)  # h /8 # c = 25
        # (N, 256, h / 4, w / 4) -> (N, 512, h / 8, w / 8)
        self.down4 = enBlock(self.g_nc * 8, self.g_nc * 8, normer)  # h / 16


        # mid layer
        self.mid1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.g_nc * 8, self.lat_dim, kernel_size=4, stride=1, padding=0, bias=False)),
            normer(self.lat_dim),# 原有的skip-Gan中没有注释的两个
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mid2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.lat_dim, self.g_nc * 8, kernel_size=4, stride=1, padding=0, bias=False)),
            normer(self.g_nc * 8),
            nn.ReLU(inplace=True)
        )

        # upconv
        self.up4 = DeBlock(self.g_nc * 16, self.g_nc * 8, normer) # output 512: 1024 -> 512
        self.up3 = DeBlock(self.g_nc * 16, self.g_nc * 4, normer) # output 512:1024 -> 512
        self.up2 = DeBlock(self.g_nc * 8, self.g_nc * 2, normer)
        # self.up1 = DeBlock(self.g_nc * 4, self.g_nc , normer)
        self.output = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.g_nc * 2, self.img_nc, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.Tanh()
        )

    def forward(self, inputs):
        x = self.down1(inputs)
        x1 = x
        # print(f"x1.shape={x1.shape}")
        x = self.down2(x)
        x2 = x
        # print(f"x2.shape={x2.shape}")
        x  = self.down3(x)
        x3 = x
        # print(f"x3.shape={x.shape}")
        x  = self.down4(x)
        x4 = x
        # print(f"x4.shape={x.shape}")

        # mid layer
        x = self.mid1(x)
        x_mid1 = x
        # print("x_mid1.shape", x_mid1.shape)
        x = self.mid2(x)
        # print("x_mid2.shape", x.shape)

        # up conv
        x = self.up4(x4, x)
        # print(f"up_x4.shape={x.shape}")
        x = self.up3(x3, x)
        # print(f"up_x3.shape={x.shape}")
        x = self.up2(x2, x)
        # print(f"up_x2.shape={x.shape}")
        # x = self.up1(x1, x)
        x = self.output(x)
        return x, x_mid1
#--------------------------------------G end--------------------------------------------------------------#

# --------------------------------------G 128--------------------------------------------------------------
class UnetSkip_RGB128(nn.Module):
    def __init__(self, opt, normer=nn.BatchNorm2d):
    # def __init__(self, img_nc=3, g_dim=64, lat_dim=100, normer=nn.BatchNorm2d):
        """
        第一层先经过一个3*3卷积，步长为1;再经过一个4×4，步长为2
        :param img_nc: image input channel:3
        :param g_nc: final output, innermost latent
        :param norm:批量归一化方式.BatchNorm2d | InStance
        """
        super(UnetSkip_RGB128, self).__init__()
        self.opt = opt
        self.img_nc = self.opt.nc
        self.g_nc = self.opt.g_dim
        self.lat_dim = self.opt.lat_dim

        # self.img_nc = img_nc
        # self.g_nc = g_dim
        # self.lat_dim = lat_dim

        # (N, 3, h, w) -> (N, 64, h/2, w/2)
        self.down1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.img_nc, self.g_nc,
                                   kernel_size=3, stride=1, padding=1, bias=False)),
            normer(self.g_nc),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(self.g_nc, self.g_nc * 2,
                                   kernel_size=4, stride=2, padding=1, bias=False)),
            normer(self.g_nc * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )  #  h / 2

        self.down1_out = self.down1

        # (N, 64, h, w) -> (N, 128, h / 2, w /  2)
        self.down2 = enBlock(self.g_nc * 2, self.g_nc * 4, normer)  # h / 4
        # (N, 128, h / 2, w / 2) -> (N, 256, h / 4, w / 4)
        self.down3 = enBlock(self.g_nc * 4, self.g_nc * 8, normer)  # h /8 # c = 25
        # (N, 256, h / 4, w / 4) -> (N, 512, h / 8, w / 8)
        self.down4 = enBlock(self.g_nc * 8, self.g_nc * 8, normer)  # h / 16
        self.down5 = enBlock(self.g_nc * 8, self.g_nc * 8, normer)  # h / 32


        # mid layer
        self.mid1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.g_nc * 8, self.lat_dim, kernel_size=4, stride=1, padding=0, bias=False)),
            normer(self.lat_dim),# 原有的skip-Gan中没有注释的两个
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mid2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.lat_dim, self.g_nc * 8, kernel_size=4, stride=1, padding=0, bias=False)),
            normer(self.g_nc * 8),
            nn.ReLU(inplace=True)
        )

        # upconv
        self.up5 = DeBlock(self.g_nc * 16, self.g_nc * 8, normer) # output 512: 1024 -> 512
        self.up4 = DeBlock(self.g_nc * 16, self.g_nc * 8, normer) # output 512: 1024 -> 512
        self.up3 = DeBlock(self.g_nc * 16, self.g_nc * 4, normer) # output 512:1024 -> 512
        self.up2 = DeBlock(self.g_nc * 8, self.g_nc * 2, normer)
        # self.up1 = DeBlock(self.g_nc * 4, self.g_nc , normer)
        self.output = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.g_nc * 2, self.img_nc, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.Tanh()
        )

    def forward(self, inputs):
        x = self.down1(inputs)
        x1 = x
        # print(f"x1.shape={x1.shape}")
        x = self.down2(x)
        x2 = x
        # print(f"x2.shape={x2.shape}")
        x  = self.down3(x)
        x3 = x
        # print(f"x3.shape={x.shape}")
        x  = self.down4(x)
        x4 = x
        # print(f"x4.shape={x.shape}")
        x = self.down5(x)
        x5 = x

        # mid layer
        x = self.mid1(x)
        x_mid1 = x
        # print("x_mid1.shape", x_mid1.shape)
        x = self.mid2(x)
        # print("x_mid2.shape", x.shape)

        # up conv
        x = self.up5(x5, x)
        x = self.up4(x4, x)
        # print(f"up_x4.shape={x.shape}")
        x = self.up3(x3, x)
        # print(f"up_x3.shape={x.shape}")
        x = self.up2(x2, x)
        # print(f"up_x2.shape={x.shape}")
        # x = self.up1(x1, x)
        x = self.output(x)
        return x, x_mid1
#--------------------------------------G end--------------------------------------------------------------#

# --------------------------------------G 256--------------------------------------------------------------
class UnetSkip_RGB256(nn.Module):
    def __init__(self, opt, normer=nn.BatchNorm2d):
    # def __init__(self, img_nc=3, g_dim=64, lat_dim=100, normer=nn.BatchNorm2d):
        """
        第一层先经过一个3*3卷积，步长为1;再经过一个4×4，步长为2
        :param img_nc: image input channel:3
        :param g_nc: final output, innermost latent
        :param norm:批量归一化方式.BatchNorm2d | InStance
        """
        super(UnetSkip_RGB256, self).__init__()
        self.opt = opt
        self.img_nc = self.opt.nc
        self.g_nc = self.opt.g_dim
        self.lat_dim = self.opt.lat_dim

        # self.img_nc = img_nc
        # self.g_nc = g_dim
        # self.lat_dim = lat_dim

        # (N, 3, h, w) -> (N, 64, h/2, w/2)
        self.down1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.img_nc, self.g_nc,
                                   kernel_size=3, stride=1, padding=1, bias=False)),
            normer(self.g_nc),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(self.g_nc, self.g_nc * 2,
                                   kernel_size=4, stride=2, padding=1, bias=False)),
            normer(self.g_nc * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )  #  h / 2

        self.down1_out = self.down1

        # (N, 64, h, w) -> (N, 128, h / 2, w /  2)
        self.down2 = enBlock(self.g_nc * 2, self.g_nc * 4, normer)  # h / 4
        # (N, 128, h / 2, w / 2) -> (N, 256, h / 4, w / 4)
        self.down3 = enBlock(self.g_nc * 4, self.g_nc * 8, normer)  # h /8 # c = 25
        # (N, 256, h / 4, w / 4) -> (N, 512, h / 8, w / 8)
        self.down4 = enBlock(self.g_nc * 8, self.g_nc * 8, normer)  # h / 16
        self.down5 = enBlock(self.g_nc * 8, self.g_nc * 8, normer)  # h / 32
        self.down6 = enBlock(self.g_nc * 8, self.g_nc * 8, normer)  # h / 32


        # mid layer
        self.mid1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.g_nc * 8, self.lat_dim, kernel_size=4, stride=1, padding=0, bias=False)),
            normer(self.lat_dim),# 原有的skip-Gan中没有注释的两个
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mid2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.lat_dim, self.g_nc * 8, kernel_size=4, stride=1, padding=0, bias=False)),
            normer(self.g_nc * 8),
            nn.ReLU(inplace=True)
        )

        # upconv
        self.up6 = DeBlock(self.g_nc * 16, self.g_nc * 8, normer) # output 512: 1024 -> 512
        self.up5 = DeBlock(self.g_nc * 16, self.g_nc * 8, normer) # output 512: 1024 -> 512
        self.up4 = DeBlock(self.g_nc * 16, self.g_nc * 8, normer) # output 512: 1024 -> 512
        self.up3 = DeBlock(self.g_nc * 16, self.g_nc * 4, normer) # output 512:1024 -> 512
        self.up2 = DeBlock(self.g_nc * 8, self.g_nc * 2, normer)
        # self.up1 = DeBlock(self.g_nc * 4, self.g_nc , normer)
        self.output = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.g_nc * 2, self.img_nc, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.Tanh()
        )

    def forward(self, inputs):
        x = self.down1(inputs)
        x1 = x
        # print(f"x1.shape={x1.shape}")
        x = self.down2(x)
        x2 = x
        # print(f"x2.shape={x2.shape}")
        x  = self.down3(x)
        x3 = x
        # print(f"x3.shape={x.shape}")
        x  = self.down4(x)
        x4 = x
        # print(f"x4.shape={x.shape}")
        x = self.down5(x)
        x5 = x

        x = self.down6(x)
        x6  = x

        # mid layer
        x = self.mid1(x)
        x_mid1 = x
        # print("x_mid1.shape", x_mid1.shape)
        x = self.mid2(x)
        # print("x_mid2.shape", x.shape)

        # up conv
        x = self.up5(x6, x)
        x = self.up5(x5, x)
        x = self.up4(x4, x)
        # print(f"up_x4.shape={x.shape}")
        x = self.up3(x3, x)
        # print(f"up_x3.shape={x.shape}")
        x = self.up2(x2, x)
        # print(f"up_x2.shape={x.shape}")
        # x = self.up1(x1, x)
        x = self.output(x)
        return x, x_mid1
#--------------------------------------G end--------------------------------------------------------------#


if __name__ == "__main__": 
    
    # gen = UnetSkip_RGB256(3, 128, 100)
    # # print(net)
    # # print(gen)
    # x = torch.randn((2, 3, 256, 256),dtype=torch.float32)
    # # out = net(x)
    # out , _= gen(x)
    
    # print(gen)
    # print(out.shape)
    # x = 128
    # res = int(np.log2(128))
    # print(f"res = {res}")
    
    # net  = Discriminator(3, 64, 64)
    # out = net(x)
    # print(out.size())
    # print(out)
    netG = GenNetwork(img_resolution=256, img_chans=3, lat_dim=100)
    data = torch.randn(2,3, 256, 256)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # print(netG)
    # out = netG(data)
    netD = Discriminator(img_resolution=256, img_chans=3)
    print('# generator parameters:', sum(param.numel() for param in netD.parameters()))
    out, fea = netD(data)
    print(out.shape)

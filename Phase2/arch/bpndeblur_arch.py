import re
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith("spade")
        parsed = re.search("spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE"
                % param_free_norm_type
            )

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, label_nc=3):
        super().__init__()
        # Attributes
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        # self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if "spectral" in opt:
            self.conv_0 = spectral_norm(self.conv_0)
            # self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.replace("spectral", "")
        self.norm_0 = SPADE(spade_config_str, fin, label_nc)
        # self.norm_1 = SPADE(spade_config_str, fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, label_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        # dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, act, kernel_size=3):
        super().__init__()
        self.act = act
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            act,
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return self.act(out)


class Down_ConvBlock(nn.Module):
    def __init__(
        self, dim_in, dim_out, activation=nn.LeakyReLU(0.2, False), kernel_size=3
    ):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size)),
            activation,
            nn.Upsample(scale_factor=1 / 2),
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size)),
            activation,
        )

    def forward(self, x):
        # conv1 = self.conv1(x)
        y = self.conv_block(x)
        return y


class KernelEncoder(nn.Module):
    def __init__(self, cin=15 * 15, wf=64):
        super(KernelEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ResnetBlock(cin, nn.LeakyReLU(0.2, False)),
            ResnetBlock(cin, nn.LeakyReLU(0.2, False)),
            ResnetBlock(cin, nn.LeakyReLU(0.2, False)),
            nn.Conv2d(cin, wf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, False),
        )
        self.conv_latent_down1 = Down_ConvBlock(wf, 2 * wf)
        self.conv_latent_down2 = Down_ConvBlock(2 * wf, 4 * wf)
        self.conv_latent_down3 = Down_ConvBlock(4 * wf, 8 * wf)

    def forward(self, z):
        latent_1 = self.encoder(z)  # 1
        latent_2 = self.conv_latent_down1(latent_1)  # 2
        latent_3 = self.conv_latent_down2(latent_2)  # 4
        latent_4 = self.conv_latent_down3(latent_3)  # 8
        # 1 2 4 8*wf
        latent_list = [latent_1, latent_2, latent_3, latent_4]
        return latent_list  # latent_6,latent_5,latent_4,latent_3,latent_2,latent_1


class UNetConvBlock(nn.Module):
    def __init__(
        self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False
    ):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_size, out_size, kernel_size=2, stride=2, bias=True
        )
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class BpnDeblur(nn.Module):

    def __init__(
        self,
        in_chn=3,
        kernel_chn=15 * 15,
        wf=32,
        depth=5,
        relu_slope=0.2,
        hin_position_left=0,
        hin_position_right=4,
    ):
        super(BpnDeblur, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.ad1_list = nn.ModuleList()
        self.KernelEncoder = KernelEncoder(cin=kernel_chn, wf=wf)

        prev_channels = wf
        # print("HINet generator normalization", opt.norm_G)
        # norm_G = "spectralspadesyncbatch3x3"
        norm_G = "spectralspadebatch3x3"
        for i in range(depth):  # 0,1,2,3,4
            use_HIN = (
                True if hin_position_left <= i and i <= hin_position_right else False
            )
            downsample = True if (i + 1) < depth else False
            self.down_path_1.append(
                UNetConvBlock(
                    prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN
                )
            )
            self.ad1_list.append(
                SPADEResnetBlock((2**i) * wf, (2**i) * wf, norm_G, label_nc=(2**i) * wf)
            )
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.ad1_list = self.ad1_list[0:-1]
        for i in reversed(range(depth - 1)):  # 3 2 1 0
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            prev_channels = (2**i) * wf

        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, kernel_field):
        with torch.no_grad():
            kernel_field = torch.flip(kernel_field, dims=[1, 2])
        # kernel_field: bijhw->b ij h w
        kernel_field = kernel_field.view(kernel_field.shape[0], -1, kernel_field.shape[-2], kernel_field.shape[-1])
        image = x
        latent_list = self.KernelEncoder(kernel_field)
        # stage 1
        x1 = self.conv_01(image)
        encs = []
        # print("x1", x1.shape)
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                # print("i--spade", i, x1.shape, latent_list[i].shape)
                # x1 = self.ad1_list[i](x1, latent_list[i])
                # print("i--spade output", i, x1.shape)
                x1, x1_up = down(x1)  # 64, 128, 128 -- 64, 256, 256
                # print("i", i, x1.shape, x1_up.shape)

                encs.append(x1_up)
            else:
                # print("i", i, x1.shape, latent_list[i].shape)
                # x1 = self.ad1_list[i](x1, latent_list[i])
                # print("i spade", i, x1.shape)
                x1 = down(x1)  # 2048, 8, 8
                # print("i - nodown", i, x1.shape)
                # x1 = self.ad1_list[-1](x1, latent_list[-1])

        for i, up in enumerate(self.up_path_1):
            # temps = self.skip_conv_1[i](encs[-i-1])
            # (8,8) ---- (1024,16,16) --- (16,16)
            # print("i temps2 input", i, encs[-i-1].shape, latent_list[-2-i].shape)
            temps2 = self.ad1_list[-1 - i](encs[-i - 1], latent_list[-1 - i])
            # print("i, temps shape", i, x1.shape, encs[-i-1].shape, temps.shape, temps2.shape)
            x1 = up(x1, temps2)
        out = self.last(x1)
        out = out + image
        return out

# @ARCH_REGISTRY.register()
# class BpnDeblur(nn.Module):

#     def __init__(
#         self,
#         in_chn=3,
#     ):
#         super(BpnDeblur, self).__init__()
#         self.last = conv3x3(3, 3, bias=True)

#     def forward(self, x, kernel_coe, kernel_basis):
#         out = self.last(x)
#         return out


# if __name__ == "__main__":
#     height = 256
#     width = 256
#     model = BpnDeblur()
#     print(model)

#     x = torch.randn((3, 3, height, width))
#     kernel = torch.randn((3, 15, 15, height, width))
#     x = model(x, kernel)
#     print(x.shape)

# def count_parameters(m):
#     return sum(p.numel() for p in m.parameters())

# model = BpnDeblur()
# total_params = count_parameters(model)
# print(f"Total parameters in the model: {total_params / 1000000} M")

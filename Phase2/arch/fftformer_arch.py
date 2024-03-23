import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numbers
from arch.bpndeblur_arch import SPADEResnetBlock
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False):
        super(TransformerBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class Fuse(nn.Module):
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerBlock(dim=n_feat * 2)

        self.blur_spade = SPADEResnetBlock(n_feat, n_feat, "spectralspadebatch3x3", n_feat)
        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc, blur_feat):
        enc = self.blur_spade(enc, blur_feat)
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.att_channel(x)
        x = self.conv2(x)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        output = e + d

        return output


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
##---------- KernelEncoder -----------------------

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


    def forward(self, z):
        latent_1 = self.encoder(z)  # 1
        latent_2 = self.conv_latent_down1(latent_1)  # 2
        latent_3 = self.conv_latent_down2(latent_2)  # 4
        # 1 2 4*wf
        latent_list = [latent_1, latent_2, latent_3]
        return latent_list

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)
##########################################################################
##---------- FFTformer -----------------------
class fftformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 4, 4, 6],
                 num_refinement_blocks=2,
                 ffn_expansion_factor=3,
                 bias=False,
                 kernel_chn=225
                 ):
        super(fftformer, self).__init__()
        self.KernelEncoder = KernelEncoder(cin=kernel_chn, wf=dim)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in
            range(num_blocks[0])])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2))

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 4))

        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 8), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[3])])
        #
        self.up4_3 = Upsample(int(dim * 8))
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 4))
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_refinement_blocks)])

        self.fuse3 = Fuse(dim * 4)
        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, inp_kernel):
    # def forward(self, inp_img):
        # kernel_field = torch.rand((inp_img.shape[0], 15, 15, inp_img.shape[-2], inp_img.shape[-1])).cuda()
        kernel_field = torch.flip(inp_kernel, dims=[1, 2])
        # kernel_field: bijhw->b ij h w
        kernel_field = kernel_field.view(kernel_field.shape[0], -1, kernel_field.shape[-2], kernel_field.shape[-1])
        kernel_latent_list = self.KernelEncoder(kernel_field)

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)

        out_dec_level4 = self.decoder_level4(inp_enc_level4)

        inp_dec_level3 = self.up4_3(out_dec_level4)
        inp_dec_level3 = self.fuse3(inp_dec_level3, out_enc_level3, kernel_latent_list[-1])
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2, kernel_latent_list[-2])
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1, kernel_latent_list[-3])
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == "__main__":
    height = 256
    width = 256
    model = fftformer().cuda()

    from ptflops import get_model_complexity_info

    with torch.no_grad():
        x = torch.randn((1, 3, height, width)).cuda()
        kernel_field = torch.randn((1, 15, 15, height, width))
        macs, params = get_model_complexity_info(model, (3, height, width), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    x = model(x)
    print(x.shape)
# def count_parameters(m):
#     return sum(p.numel() for p in m.parameters())

# model = fftformer()
# total_params = count_parameters(model)
# print(f"Total parameters in the model: {total_params / 1000000} M")


# if __name__ == "__main__":
#     height = 256
#     width = 256
#     model = fftformer()
#     print(model)

#     x = torch.randn((4, 3, height, width))
#     kernel_field = torch.randn((4, 15, 15, height, width))
#     x = model(x, kernel_field)
#     print(x.shape)
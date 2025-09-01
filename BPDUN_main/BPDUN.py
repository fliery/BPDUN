
import torch.nn as nn
import torch
from DWT_IDWT.DWT_IDWT_layer import *
from einops.layers.torch import Rearrange

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None, pad_model=None, groups=1):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

        if self.pad_model == None:
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding,
                                        groups=groups, bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,
                                        groups=groups, bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class BSConvU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, dilation=1, bias=True,
                 padding_mode="zeros", with_bn=False, bn_kwargs=None):
        super().__init__()
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}
        # pointwise
        self.add_module("pw", torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))
        # batchnorm
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))
        # depthwise
        self.add_module("dw", torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        ))

class DEUM(nn.Module):
    def __init__(self, in_channels):
        super(DEUM, self).__init__()
        self.conv4 = BSConvU(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = BSConvU(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv6 = BSConvU(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ms, ph1, ph2, ph3):
        ms1 = ms
        PH1_attn = torch.mul(self.sigmoid(self.conv4(ms1)), ph1)
        PH2_attn = torch.mul(self.sigmoid(self.conv5(ms1)), ph2)
        PH3_attn = torch.mul(self.sigmoid(self.conv6(ms1)), ph3)

        return PH1_attn, PH2_attn, PH3_attn

class IRB(nn.Module):
    def __init__(self, mschannel, lam=0.7):
        super(IRB, self).__init__()
        self.xita = nn.Parameter(torch.tensor(lam))
        self.blur = ConvBlock(mschannel, mschannel, 3, 1, 1, activation='prelu', norm=None, bias=False)
        self.inverblur = ConvBlock(mschannel, mschannel, 3, 1, 1, activation='prelu', norm=None, bias=False)

    def forward(self, hrms, M, lrms):
        g = self.inverblur(self.blur(hrms)-lrms) + self.xita * (M - hrms)
        return g

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class BS(nn.Module):
    def __init__(self, channels):
        super(BS, self).__init__()
        self.BSConv3 = BSConvU(channels, channels, kernel_size=3, padding=1)
        self.BSConv5 = BSConvU(channels, channels, kernel_size=5, padding=2)
        self.BSConv7 = BSConvU(channels, channels, kernel_size=7, padding=3)
        self.conv1 = BSConvU(channels, channels, kernel_size=3, padding=1)
        self.conv1_down = nn.Conv2d(in_channels=channels * 4, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.GELU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.pa = PixelAttention(channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.mul(self.pa(x, self.sigmoid(self.conv1(x))), self.GELU(self.BSConv3(x))) # CGPAM
        out1 = torch.add(x1, x2)
        x3 = self.conv1(out1)
        x4 = torch.mul(self.pa(out1,self.sigmoid(self.conv1(out1))), self.GELU(self.BSConv5(out1))) # CGPAM
        out2 = torch.add(x3, x4)
        x5 = self.conv1(out2)
        x6 = torch.mul(self.pa(out2,self.sigmoid(self.conv1(out2))), self.GELU(self.BSConv7(out2))) # CGPAM
        out = self.conv1_down(torch.cat((x1, x3, x5, x6), 1))
        return out

class BF(nn.Module):
    def __init__(self, channels):
        super(BF, self).__init__()
        self.BSConv3 = BSConvU(channels, channels, kernel_size=3, padding=1)
        self.BSConv5 = BSConvU(channels, channels, kernel_size=5, padding=2)
        self.BSConv7 = BSConvU(channels, channels, kernel_size=7, padding=3)
        self.conv1 = BSConvU(channels, channels, kernel_size=3, padding=1)
        self.conv1_down = nn.Conv2d(in_channels=channels * 4, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.GELU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.pa = PixelAttention(channels)
        self.conv2_down = nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=1, stride=1,
                                    padding=0)

    def forward(self, x0):
        x00 = self.conv2_down(x0)
        _, _, H, W = x0.shape
        dim = 1
        y = torch.fft.rfft2(x00, norm='backward')
        y_imag = y.imag
        y_real = y.real
        x = torch.cat([y_real, y_imag], dim=dim)
        x1 = self.conv1(x)
        x2 = torch.mul(self.pa(x, self.sigmoid(self.conv1(x))), self.GELU(self.BSConv3(x))) # CGPAM
        out1 = torch.add(x1, x2)
        x3 = self.conv1(out1)
        x4 = torch.mul(self.pa(out1,self.sigmoid(self.conv1(out1))), self.GELU(self.BSConv5(out1))) # CGPAM
        out2 = torch.add(x3, x4)
        x5 = self.conv1(out2)
        x6 = torch.mul(self.pa(out2,self.sigmoid(self.conv1(out2))), self.GELU(self.BSConv7(out2))) # CGPAM
        out = self.conv1_down(torch.cat((x1, x3, x5, x6), 1))
        y_real1, y_imag1 = torch.chunk(out, 2, dim=dim)
        y = torch.complex(y_real1, y_imag1)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')

        return y

class Branch2(nn.Module):
    def __init__(self, channels):
        super(Branch2, self).__init__()

        self.BranchS = BS(channels)   # spatial
        self.BranchF = BF(channels)  # frequency
        self.down = nn.Conv2d(in_channels=channels // 2 * 3, out_channels=channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x0):
        x1 = self.BranchS(x0)
        x2 = self.BranchF(x0)
        x = torch.concat((x1, x2),dim=1)
        x = self.down(x)
        return x


# for 4 bands
class PDB(nn.Module):
    def __init__(self, mschannel, mid):
        super(PDB, self).__init__()
        self.base = Branch2(mid)
        self.final = ConvBlock(mid, mschannel, kernel_size=3, stride=1, padding=1, bias=True,activation='prelu')
        self.conv11 = ConvBlock(2, mid, 3, 1, 1, activation='prelu', norm=None, bias=False)
        self.conv12 = ConvBlock(2, mid, 3, 1, 1, activation='prelu', norm=None, bias=False)
        self.conv13 = ConvBlock(2, mid, 3, 1, 1, activation='prelu', norm=None, bias=False)
        self.conv14 = ConvBlock(2, mid, 3, 1, 1, activation='prelu', norm=None, bias=False)
        self.conv19 = ConvBlock(mid * 4, mid, 3, 1, 1, activation='prelu', norm=None, bias=False)

    def forward(self, hrms1, pan):
        S = hrms1
        s1 = S[:, 0, :, :].unsqueeze(1)  # for 4 bands
        s2 = S[:, 1, :, :].unsqueeze(1)
        s3 = S[:, 2, :, :].unsqueeze(1)
        s4 = S[:, 3, :, :].unsqueeze(1)
        x1 = torch.cat((s1, pan), dim=1)
        x2 = torch.cat((s2, pan), dim=1)
        x3 = torch.cat((s3, pan), dim=1)
        x4 = torch.cat((s4, pan), dim=1)
        x1 = self.conv11(x1)
        x2 = self.conv12(x2)
        x3 = self.conv13(x3)
        x4 = self.conv14(x4)
        x9 = torch.cat((x1, x2, x3, x4), dim=1)
        x9 = self.conv19(x9)
        x2 = self.base(x9)
        out = self.final(x2) + hrms1
        return out


class BPDUN(nn.Module):
    def __init__(self, mschannel=4, stage1=2, stage2=2, stage3=2, mid=48, lam=0.7, eta=1.0, wavename='haar'):
        super(BPDUN, self).__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.mschannel = mschannel

        self.dwt0 = DWT_2D(wavename=wavename)
        self.dwt1 = DWT_2D(wavename=wavename)
        self.idwt0 = IDWT_2D(wavename=wavename)
        self.idwt1 = IDWT_2D(wavename=wavename)

        self.updatems1 = IRB(mschannel=mschannel, lam = lam)
        self.updatems2 = IRB(mschannel=mschannel, lam = lam)
        self.updatems3 = IRB(mschannel=mschannel, lam = lam)

        self.proxM1 = PDB(mschannel=mschannel, mid=mid)
        self.proxM2 = PDB(mschannel=mschannel, mid=mid)
        self.proxM3 = PDB(mschannel=mschannel, mid=mid)

        self.cdhfm1 = DEUM(mschannel)
        self.cdhfm2 = DEUM(mschannel)


        self.eta1 = nn.ParameterList([nn.Parameter(torch.tensor(eta), requires_grad=True) for _ in range(stage1)])
        self.eta2 = nn.ParameterList([nn.Parameter(torch.tensor(eta), requires_grad=True) for _ in range(stage2)])
        self.eta3 = nn.ParameterList([nn.Parameter(torch.tensor(eta), requires_grad=True) for _ in range(stage3)])

        print('stage123: ', self.stage1,self.stage2,self.stage3)

    def forward(self, lrms, pan):
        upan = pan.repeat_interleave(self.mschannel,dim=1)
        pd_0, pg_01, pg_02, pg_03 = self.dwt0(upan)  # LL, LH, HL, HH
        pd_1, pg_11, pg_12, pg_13 = self.dwt1(pd_0)   # LL, LH, HL, HH

        # stage 1
        ms1 = lrms
        pd11 = pd_1[:,0,:,:].unsqueeze(1)
        for i in range(self.stage1):
            M1 = self.proxM1(ms1,  pd11)
            ms1 = ms1 - self.eta1[i]*self.updatems1(ms1, M1, lrms)
        pg_11, pg_12, pg_13 = self.cdhfm1(ms1, pg_11, pg_12, pg_13)
        ms20 = self.idwt1(ms1,pg_11, pg_12, pg_13)

        # stage 2
        ms2 = ms20
        pd00 = pd_0[:,0,:,:].unsqueeze(1)
        for i in range(self.stage2):
            M2 = self.proxM2(ms2, pd00)
            ms2 = ms2 - self.eta2[i] * self.updatems2(ms2, M2, ms20)
        pg_01, pg_02, pg_03 = self.cdhfm2(ms2, pg_01, pg_02, pg_03)
        ms30 = self.idwt0(ms2, pg_01, pg_02, pg_03)

        # stage 3
        ms3 = ms30
        for i in range(self.stage3):
            M3 = self.proxM3(ms3, pan)
            ms3 = ms3 - self.eta3[i] * self.updatems3(ms3, M3, ms30)
        out = ms3

        return out


if __name__ == '__main__':
    import time
    from thop import profile
    input1 = torch.randn(1, 4, 32, 32).to(torch.device('cuda:0'))
    input2 = torch.randn(1, 1, 128, 128).to(torch.device('cuda:0'))
    prev_time = time.time()
    model = BPDUN(4,2,2,2,48).to(torch.device('cuda:0'))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    s = time.time()
    with torch.no_grad():
        c = model(input1, input2)
    gpu_memory_used = torch.cuda.max_memory_allocated(device=torch.device('cuda:0'))
    e = time.time()
    print('one fuse testing time(s): ', str(e - s))
    flops, params = profile(model, inputs=(input1,input2, ))
    print('FLOPs = ' + str(flops/1e9) + 'G')
    print('Params = ' + str(params/1e6) + 'M')
    gpu_memory_used_mb = gpu_memory_used / (1024 ** 2)  # MB
    print(f"GPU memory used during inference: {gpu_memory_used_mb:.4f} MB")



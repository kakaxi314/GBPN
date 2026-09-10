from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
from collections import OrderedDict
import torch.distributed as dist
from timm.models.layers import DropPath
import GbpnOps

__all__ = [
    'EMA',
    'inplace_relu',
    'weights_init',
    'Conv1x1',
    'Conv3x3',
    'BasicBlock',
    'Basic2dTrans',
    'Basic2d',
    'GBP',
]


class EMA(nn.Module):
    """ Model Exponential Moving Average V2 borrow from timm https://timm.fast.ai/

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, start_epoch=0, ddp=False):
        super().__init__()
        self.start_epoch = start_epoch
        self.module = deepcopy(model)
        self.module.eval()
        self.zeros()
        self.iter = 0
        if ddp:
            self.broadcast()
        self.decay = decay

    def zeros(self):
        with torch.no_grad():
            for ema_v in self.module.state_dict().values():
                ema_v.zero_()

    def broadcast(self):
        for ema_v in self.module.state_dict().values():
            dist.broadcast(ema_v, src=0, async_op=False)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, epoch=0):
        if epoch >= self.start_epoch:
            self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
            self.iter += 1

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def get(self):
        net = deepcopy(self.module)
        scale = 1 - self.decay ** self.iter
        with torch.no_grad():
            for ema_v in net.state_dict().values():
                ema_v.copy_(ema_v / scale)
        return net


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def weights_init(m, mode='trunc'):
    from torch.nn.init import _calculate_fan_in_and_fan_out
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if hasattr(m, 'weight'):
            if mode == 'trunc':
                torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=0.02)
            elif mode == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data)
            else:
                raise ValueError(f'unknown mode = {mode}')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Conv1d') != -1:
        if hasattr(m, 'weight'):
            if mode == 'trunc':
                torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=0.02)
            elif mode == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data)
            else:
                raise ValueError(f'unknown mode = {mode}')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if mode == 'trunc':
            torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=.02)
        elif mode == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data)
        else:
            raise ValueError(f'unknown mode = {mode}')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


def Conv1x1(in_planes, out_planes, stride=1, bias=False, groups=1, dilation=1, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding_mode='zeros', bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, padding_mode=padding_mode, groups=groups, bias=bias, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, padding_mode='zeros', act=nn.ReLU,
                 last=True, drop_path=0.0):
        super().__init__()
        bias = False
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        self.conv1 = Conv3x3(inplanes, planes, stride, padding_mode=padding_mode, bias=bias)
        self.bn1 = norm_layer(planes)
        self.relu1 = act()
        self.conv2 = Conv3x3(planes, planes, padding_mode=padding_mode, bias=bias)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if last:
            self.relu2 = act()

    def forward(self, x, d=None):
        identity = x
        if d is not None:
            x = torch.cat([x, d], dim=1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.drop_path(out) + identity
        if self.last:
            out = self.relu2(out)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, act=nn.ReLU):
        super().__init__()
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2,
                                       stride=2, padding=0, bias=bias)
        self.bn = norm_layer(out_channels)
        self.relu = act()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1, padding_mode='zeros',
                 act=nn.ReLU, stride=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False, padding_mode=padding_mode)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv = nn.Sequential(OrderedDict([('conv', conv)]))
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', act())

    def forward(self, x):
        out = self.conv(x)
        return out


class UpCatIden(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1,
                 padding_mode='replicate', act=nn.ReLU):
        super().__init__()
        self.upf = Basic2dTrans(in_channels, out_channels, norm_layer=norm_layer, act=nn.Identity)
        self.conv = Basic2d(out_channels + out_channels, out_channels,
                            norm_layer=norm_layer, kernel_size=kernel_size,
                            padding=padding, padding_mode=padding_mode, act=nn.Identity)

    def forward(self, x, y):
        """
        """
        out = self.upf(x)
        fout = torch.cat([out, y], dim=1)
        fout = self.conv(fout)
        return fout


class kaGBPLR(Function):
    """
    """

    @staticmethod
    def forward(ctx, input, prior, beta, post, mask, mprob, prob):
        B, C, H, W = input.shape
        output = input.clone().contiguous()
        conf = prior.clone().contiguous()
        output9 = (output.sum(dim=1, keepdim=True) + mprob).contiguous()
        conf9 = (conf.sum(dim=1, keepdim=True) + prob).contiguous()
        for j in range(W):
            GbpnOps.GBPLR_F(beta, post, mask, output, conf, output9, conf9, j)
        ctx.save_for_backward(input, prior, beta, post, mask, output, conf, output9, conf9)
        return output, conf

    @staticmethod
    def backward(ctx, dout, dconf):
        B, _, H, W = dout.shape
        dout = dout.clone().contiguous()
        dconf = dconf.clone().contiguous()
        input, prior, beta, post, mask, output, conf, output9, conf9 = ctx.saved_tensors
        dout9 = torch.zeros_like(output9)
        dconf9 = torch.zeros_like(conf9)
        dbeta = torch.zeros_like(beta)
        dpost = torch.zeros_like(post)
        dmask = torch.zeros_like(mask)
        for j in range(W - 1, -1, -1):
            GbpnOps.GBPLR_B(dout, dconf, dout9, dconf9, dbeta, dpost, dmask,
                            input, prior, beta, post, mask, output, conf, output9, conf9, j)
        dinput = dout + dout9
        dprior = dconf + dconf9
        return dinput, dprior, dbeta, dpost, dmask, dout9, dconf9


class kaGBPRL(Function):

    @staticmethod
    def forward(ctx, input, prior, beta, post, mask, mprob, prob):
        B, C, H, W = input.shape
        output = input.clone().contiguous()
        conf = prior.clone().contiguous()
        output9 = (output.sum(dim=1, keepdim=True) + mprob).contiguous()
        conf9 = (conf.sum(dim=1, keepdim=True) + prob).contiguous()
        for j in range(W - 1, -1, -1):
            GbpnOps.GBPRL_F(beta, post, mask, output, conf, output9, conf9, j)
        ctx.save_for_backward(input, prior, beta, post, mask, output, conf, output9, conf9)
        return output, conf

    @staticmethod
    def backward(ctx, dout, dconf):
        B, _, H, W = dout.shape
        dout = dout.clone().contiguous()
        dconf = dconf.clone().contiguous()
        input, prior, beta, post, mask, output, conf, output9, conf9 = ctx.saved_tensors
        dout9 = torch.zeros_like(output9)
        dconf9 = torch.zeros_like(conf9)
        dbeta = torch.zeros_like(beta)
        dpost = torch.zeros_like(post)
        dmask = torch.zeros_like(mask)
        for j in range(W):
            GbpnOps.GBPRL_B(dout, dconf, dout9, dconf9, dbeta, dpost, dmask,
                            input, prior, beta, post, mask, output, conf, output9, conf9, j)
        dinput = dout + dout9
        dprior = dconf + dconf9
        return dinput, dprior, dbeta, dpost, dmask, dout9, dconf9


class kaGBPTD(Function):

    @staticmethod
    def forward(ctx, input, prior, beta, post, mask, mprob, prob):
        B, C, H, W = input.shape
        output = input.clone().contiguous()
        conf = prior.clone().contiguous()
        output9 = (output.sum(dim=1, keepdim=True) + mprob).contiguous()
        conf9 = (conf.sum(dim=1, keepdim=True) + prob).contiguous()
        for i in range(H):
            GbpnOps.GBPTD_F(beta, post, mask, output, conf, output9, conf9, i)
        ctx.save_for_backward(input, prior, beta, post, mask, output, conf, output9, conf9)
        return output, conf

    @staticmethod
    def backward(ctx, dout, dconf):
        B, _, H, W = dout.shape
        dout = dout.clone().contiguous()
        dconf = dconf.clone().contiguous()
        input, prior, beta, post, mask, output, conf, output9, conf9 = ctx.saved_tensors
        dout9 = torch.zeros_like(output9)
        dconf9 = torch.zeros_like(conf9)
        dbeta = torch.zeros_like(beta)
        dpost = torch.zeros_like(post)
        dmask = torch.zeros_like(mask)
        for i in range(H - 1, -1, -1):
            GbpnOps.GBPTD_B(dout, dconf, dout9, dconf9, dbeta, dpost, dmask,
                            input, prior, beta, post, mask, output, conf, output9, conf9, i)
        dinput = dout + dout9
        dprior = dconf + dconf9
        return dinput, dprior, dbeta, dpost, dmask, dout9, dconf9


class kaGBPDT(Function):

    @staticmethod
    def forward(ctx, input, prior, beta, post, mask, mprob, prob):
        B, C, H, W = input.shape
        output = input.clone().contiguous()
        conf = prior.clone().contiguous()
        output9 = (output.sum(dim=1, keepdim=True) + mprob).contiguous()
        conf9 = (conf.sum(dim=1, keepdim=True) + prob).contiguous()
        for i in range(H - 1, -1, -1):
            GbpnOps.GBPDT_F(beta, post, mask, output, conf, output9, conf9, i)
        ctx.save_for_backward(input, prior, beta, post, mask, output, conf, output9, conf9)
        return output, conf

    @staticmethod
    def backward(ctx, dout, dconf):
        B, _, H, W = dout.shape
        dout = dout.clone().contiguous()
        dconf = dconf.clone().contiguous()
        input, prior, beta, post, mask, output, conf, output9, conf9 = ctx.saved_tensors
        dout9 = torch.zeros_like(output9)
        dconf9 = torch.zeros_like(conf9)
        dbeta = torch.zeros_like(beta)
        dpost = torch.zeros_like(post)
        dmask = torch.zeros_like(mask)
        for i in range(H):
            GbpnOps.GBPDT_B(dout, dconf, dout9, dconf9, dbeta, dpost, dmask,
                            input, prior, beta, post, mask, output, conf, output9, conf9, i)
        dinput = dout + dout9
        dprior = dconf + dconf9
        return dinput, dprior, dbeta, dpost, dmask, dout9, dconf9


class kaGBPNL(Function):
    @staticmethod
    def forward(ctx, input, prior, beta, post, offs):
        input = input.contiguous()
        prior = prior.contiguous()
        output = torch.zeros_like(input)
        conf = torch.zeros_like(prior)
        GbpnOps.GBPNL_F(input, prior, beta, post, offs, output, conf)
        ctx.save_for_backward(input, prior, beta, post, offs, output, conf)
        return output, conf

    @staticmethod
    def backward(ctx, dout, dconf):
        B, _, H, W = dout.shape
        dout = dout.contiguous()
        dconf = dconf.contiguous()
        input, prior, beta, post, offs, output, conf = ctx.saved_tensors
        dinput = torch.zeros_like(input)
        dprior = torch.zeros_like(prior)
        dbeta = torch.zeros_like(beta)
        dpost = torch.zeros_like(post)
        doffs = torch.zeros_like(offs)
        GbpnOps.GBPNL_B(dout, dconf, dinput, dprior, dbeta, dpost, doffs,
                        input, prior, beta, post, offs, output, conf)
        return dinput, dprior, dbeta, dpost, doffs


ka_gbplr = kaGBPLR.apply
ka_gbptd = kaGBPTD.apply
ka_gbprl = kaGBPRL.apply
ka_gbpdt = kaGBPDT.apply
ka_gbpnl = kaGBPNL.apply


def compiled_loop_lr(beta, post, mask, output, conf, output_sum, conf_sum):
    B, C, H, W = output.shape
    for j in range(W):
        GbpnOps.GBPLR_F(beta, post, mask, output, conf, output_sum, conf_sum, j)
    return output, conf


def compiled_loop_td(diff, post, beta, output, conf, output9, conf_sum):
    B, C, H, W = output.shape
    for i in range(H):
        GbpnOps.GBPTD_F(diff, post, beta, output, conf, output9, conf_sum, i)
    return output, conf


def compiled_loop_rl(diff, post, beta, output, conf, output_sum, conf_sum):
    B, C, H, W = output.shape
    for j in range(W - 1, -1, -1):
        GbpnOps.GBPRL_F(diff, post, beta, output, conf, output_sum, conf_sum, j)
    return output, conf


def compiled_loop_dt(diff, post, beta, output, conf, output_sum, conf_sum):
    B, C, H, W = output.shape
    for i in range(H - 1, -1, -1):
        GbpnOps.GBPDT_F(diff, post, beta, output, conf, output_sum, conf_sum, i)
    return output, conf


def compiled_loop_nl(input, prior, mprob, prob, diff, post, off, output, conf, beta, inv_beta, eta_curr, conf_curr):
    for i in range(4):
        output.zero_()
        conf.zero_()
        torch.add(mprob, eta_curr, out=input)
        torch.add(prob, conf_curr, out=prior)

        GbpnOps.GBPNL_F(input, prior, diff, post, off, output, conf)
        eta_curr.mul_(beta)
        eta_curr.addcmul_(inv_beta, output)

        conf_curr.mul_(beta)
        conf_curr.addcmul_(inv_beta, conf)
    return eta_curr, conf_curr


def PropG(prop, input, prior, diff, post, beta, mprob, prob):
    eta_out_l = input.clone().contiguous()
    conf_out_l = prior.clone().contiguous()
    output_sum = (eta_out_l.sum(dim=1, keepdim=True) + mprob).contiguous()
    conf_sum = (conf_out_l.sum(dim=1, keepdim=True) + prob).contiguous()
    eta_out_l, conf_out_l = prop(diff, post, beta, eta_out_l, conf_out_l, output_sum, conf_sum)
    return eta_out_l, conf_out_l


def PropL(prop, mprob, eta_out_nl, prob, conf_out_nl, diff, post, off, beta):
    mprob = mprob.contiguous()
    prob = prob.contiguous()

    output = torch.zeros_like(mprob)
    conf = torch.zeros_like(prob)

    input = torch.zeros_like(mprob)
    prior = torch.zeros_like(prob)

    eta_curr = eta_out_nl.clone()
    conf_curr = conf_out_nl.clone()
    inv_beta = 1.0 - beta

    eta_out_nl, conf_out_nl = prop(input, prior, mprob, prob, diff, post, off, output, conf, beta,
                                   inv_beta,
                                   eta_curr, conf_curr)
    return eta_out_nl, conf_out_nl


class GBP(nn.Module):
    """
   add beta to confps
    """

    def __init__(self, eps=1e-10, pg=3, pl=6, pp=4, height=256,
                 width=320, batch=1):
        super().__init__()
        pd = 4
        self.pd = pd
        self.pg = pg
        self.pl = pl
        self.pp = pp
        self.eps = eps
        self.height, self.width, self.batch = height, width, batch
        self.propG_train = [ka_gbplr, ka_gbptd, ka_gbprl, ka_gbpdt]
        self.propL_train = ka_gbpnl

        inputsG = tuple(
            torch.rand(self.batch, c, self.height, self.width, device="cuda") for c in [3, 3, 1, 8, 8, 1, 1])
        self.propG_test = [
            torch.cuda.make_graphed_callables(direction, inputsG)
            for direction in [compiled_loop_lr, compiled_loop_td, compiled_loop_rl, compiled_loop_dt]
        ]
        inputsL = tuple(torch.rand(self.batch, c, self.height, self.width, device="cuda") for c in
                        [1, 1, 1, 1, self.pp, self.pp, self.pp * 2, 1, 1, 1, 1, 1, 1])
        self.propL_test = torch.cuda.make_graphed_callables(compiled_loop_nl, inputsL)

    def prop_test(self, mprior, prior, conf_out_l, conf_out_nl, eta_out_l, eta_out_nl, posts, diffs, betas, offs):
        """
        """
        for i in range(self.pg):
            mprob = mprior + eta_out_nl
            prob = prior + conf_out_nl
            for j in range(self.pd):
                eta_out_l, conf_out_l = PropG(self.propG_test[j], eta_out_l, conf_out_l, diffs[i * (self.pd + 1) + j],
                                              posts[i * (self.pd + 1) + j], betas[2 * i],
                                              mprob, prob)
            mprob = mprior + eta_out_l.sum(dim=1, keepdim=True)
            prob = prior + conf_out_l.sum(dim=1, keepdim=True)
            eta_out_nl, conf_out_nl = PropL(self.propL_test, mprob, eta_out_nl, prob, conf_out_nl,
                                            diffs[i * (self.pd + 1) + self.pd],
                                            posts[i * (self.pd + 1) + self.pd], offs[i],
                                            betas[2 * i + 1])
        return conf_out_nl, eta_out_nl, eta_out_l, conf_out_l

    def prop_train(self, mprior, prior, conf_out_l, conf_out_nl, eta_out_l, eta_out_nl, posts, diffs, betas, offs):
        """
        """
        for i in range(self.pg):
            mprob = mprior + eta_out_nl
            prob = prior + conf_out_nl
            for j in range(self.pd):
                eta_out_l, conf_out_l = self.propG_train[j](eta_out_l, conf_out_l, diffs[i * (self.pd + 1) + j],
                                                            posts[i * (self.pd + 1) + j], betas[2 * i],
                                                            mprob, prob)
            mprob = mprior + eta_out_l.sum(dim=1, keepdim=True)
            prob = prior + conf_out_l.sum(dim=1, keepdim=True)
            for j in range(self.pl):
                eta_out_nl_curr, conf_out_nl_curr = self.propL_train(mprob + eta_out_nl, prob + conf_out_nl,
                                                                     diffs[i * (self.pd + 1) + self.pd],
                                                                     posts[i * (self.pd + 1) + self.pd], offs[i])
                eta_out_nl = eta_out_nl * betas[2 * i + 1] + (1 - betas[2 * i + 1]) * eta_out_nl_curr
                conf_out_nl = conf_out_nl * betas[2 * i + 1] + (1 - betas[2 * i + 1]) * conf_out_nl_curr
        return conf_out_nl, eta_out_nl, eta_out_l, conf_out_l

    def forward(self, fout, S, conf_out, eta_out):
        probability, diffs, offs, betas = fout
        conf_out_l, conf_out_nl = conf_out
        eta_out_l, eta_out_nl = eta_out
        mask = (S > 0).float()
        probability = torch.split(probability, [1, 1] + self.pg * (self.pd * [3, ] + [self.pp, ]), dim=1)
        prior = probability[0]
        conf_r = probability[1]
        posts = [(F.softplus(post) + self.eps).contiguous() for post in probability[2:]]
        prior = (F.softplus(prior) + self.eps) * mask
        mprior = S * prior
        diffs = torch.split(diffs, self.pg * (self.pd * [3, ] + [self.pp, ]), dim=1)
        diffs = [diff.contiguous() for diff in diffs]
        betas = torch.split(betas, 1, dim=1)
        betas = [beta.contiguous() for beta in betas]
        offs = torch.split(offs, 2 * self.pp, dim=1)
        offs = [off.contiguous() for off in offs]
        if self.training:
            conf_out_nl, eta_out_nl, eta_out_l, conf_out_l = self.prop_train(mprior, prior, conf_out_l, conf_out_nl,
                                                                             eta_out_l,
                                                                             eta_out_nl, posts, diffs, betas, offs)
        else:
            conf_out_nl, eta_out_nl, eta_out_l, conf_out_l = self.prop_test(mprior, prior, conf_out_l, conf_out_nl,
                                                                            eta_out_l,
                                                                            eta_out_nl, posts, diffs, betas, offs)
        conf = torch.sum(conf_out_l, dim=1, keepdim=True) + conf_out_nl + prior
        eta = torch.sum(eta_out_l, dim=1, keepdim=True) + eta_out_nl + mprior
        depth = eta / conf
        conf = conf_r + conf
        eta_out = [eta_out_l, eta_out_nl]
        conf_out = [conf_out_l, conf_out_nl]
        return eta_out, conf_out, depth, conf

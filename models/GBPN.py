from functools import partial
import torch
from click.core import batch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn, Tensor
from typing import Optional
import numpy as np
import functools
from einops import rearrange
from natten.functional import na2d_qk, na2d_av
from .utils import Conv1x1, Basic2d, BasicBlock, weights_init, inplace_relu, UpCatIden, Basic2dTrans, GBP

__all__ = [
    'GBPN',
]


class FUSE(nn.Module):
    """
    Self-attention and cross-attention fusion with Relative Position Bias (RPB),
    residual connection, drop path, and gating mechanisms.
    """

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            dilation: int = 1,
            is_causal: bool = False,
            rel_pos_bias: bool = True,
            qkv_bias: bool = True,
            qk_scale: Optional[float] = None,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer=nn.BatchNorm2d,
            hidden_scale=1,
            padding_mode='replicate',
            act=nn.ReLU,
            fuse: bool = False,
            drop_path=0.1
    ):
        super().__init__()
        self.head_dim = 32
        self.dim = dim
        self.num_heads = dim // self.head_dim
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.is_causal = is_causal
        self.qgs = Basic2d(dim, dim + self.num_heads, kernel_size=1, padding=0, norm_layer=norm_layer, act=nn.Identity)
        self.kvs = Basic2d(dim, 2 * dim, kernel_size=1, padding=0, norm_layer=norm_layer, act=nn.Identity)
        if rel_pos_bias:
            self.rpbs = nn.Parameter(
                torch.zeros(
                    self.num_heads,
                    (2 * self.kernel_size - 1),
                    (2 * self.kernel_size - 1),
                )
            )
            trunc_normal_(self.rpbs, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpbs", None)
        self.projs = Conv1x1(dim, dim)
        if fuse:
            self.qg = Basic2d(dim, dim + self.num_heads, kernel_size=1, padding=0, norm_layer=norm_layer,
                              act=nn.Identity)
            self.kv = Basic2d(dim, 2 * dim, kernel_size=1, padding=0, norm_layer=norm_layer, act=nn.Identity)
            if rel_pos_bias:
                self.rpb = nn.Parameter(
                    torch.zeros(
                        self.num_heads,
                        (2 * self.kernel_size - 1),
                        (2 * self.kernel_size - 1),
                    )
                )
                trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
            else:
                self.register_parameter("rpb", None)
            self.proj = Conv1x1(dim, dim)
            self.resblock = BasicBlock(dim, dim, norm_layer=norm_layer, padding_mode=padding_mode, drop_path=drop_path,
                                       last=False, act=nn.GELU)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.resblocks = BasicBlock(dim, dim, norm_layer=norm_layer, padding_mode=padding_mode, drop_path=drop_path,
                                    last=False, act=nn.GELU)
        self.drop_paths = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y) -> Tensor:
        if y is not None:
            q, g = torch.split(self.qg(x), [self.dim, self.num_heads], dim=1)
            q = (self.scale * rearrange(q, 'B (h d) H W -> B h H W d', d=32)).contiguous()
            g = rearrange(torch.sigmoid(g), 'B h H W -> B h H W 1')
            kv = rearrange(self.kv(y), 'B (n h d) H W-> n B h H W d', n=2, d=32).contiguous()
            k, v = kv[0], kv[1]
            attn = na2d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            xn = na2d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            xn = rearrange(xn * g, 'B h H W C -> B (h C) H W')
            x = x + self.drop_path(self.proj(xn))
            x = self.resblock(x)

        q, g = torch.split(self.qgs(x), [self.dim, self.num_heads], dim=1)
        q = (self.scale * rearrange(q, 'B (h d) H W -> B h H W d', d=32)).contiguous()
        g = rearrange(torch.sigmoid(g), 'B h H W -> B h H W 1')
        kv = rearrange(self.kvs(x), 'B (n h d) H W-> n B h H W d', n=2, d=32).contiguous()
        k, v = kv[0], kv[1]
        attn = na2d_qk(
            q,
            k,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
            rpb=self.rpbs,
        )
        attn = attn.softmax(dim=-1)
        xn = na2d_av(
            attn,
            v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
        )
        xn = rearrange(xn * g, 'B h H W C -> B (h C) H W')
        x = x + self.drop_paths(self.projs(xn))
        x = self.resblocks(x)
        return x


class UNet(nn.Module):
    """
    Multi-scale UNet backbone integrated with FUSE guidance modules to output parameters for Gaussian Belief Propagation (GBP).
    """

    def __init__(self, block=BasicBlock, bc=16, img_layers=[2, 2, 2, 2, 2],
                 dilations=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], up_layer=UpCatIden,
                 prop_layer=GBP, drop_path=0.1, pp=3, pg=1, pl=4, pd=4,
                 norm_layer=nn.BatchNorm2d,
                 padding_mode='replicate', act=nn.ReLU, guide=FUSE, inch=3, fbc=16):
        super().__init__()
        self._norm_layer = norm_layer
        self._padding_mode = padding_mode
        _img_dpc = 0
        _img_dprs = np.linspace(0, drop_path, sum(img_layers))
        self.act = act

        self.inplanes = bc * 2
        self.conv_img = nn.Sequential(
            Basic2d(inch, bc * 2, norm_layer=norm_layer, kernel_size=3, padding=1,
                    padding_mode=padding_mode, act=nn.GELU),
            block(bc * 2, bc * 2, norm_layer=norm_layer, padding_mode=padding_mode,
                  last=False, act=nn.GELU))

        self.layer0_img = Basic2d(bc * 2, bc * 4, norm_layer=norm_layer, kernel_size=3, padding=1, stride=2,
                                  act=nn.Identity)
        self.layer1_img = Basic2d(bc * 4, bc * 8, norm_layer=norm_layer, kernel_size=3, padding=1, stride=2,
                                  act=nn.Identity)
        self.layer2_img = Basic2d(bc * 8, bc * 16, norm_layer=norm_layer, kernel_size=3, padding=1, stride=2,
                                  act=nn.Identity)
        self.layer3_img = Basic2d(bc * 16, bc * 16, norm_layer=norm_layer, kernel_size=3, padding=1, stride=2,
                                  act=nn.Identity)
        self.layer4_img = Basic2d(bc * 16, bc * 16, norm_layer=norm_layer, kernel_size=3, padding=1, stride=2,
                                  act=nn.Identity)

        # self.layer0d_img = up_layer(bc * 4, bc * 2, norm_layer=norm_layer, act=act)
        self.layer1d_img = up_layer(bc * 8, bc * 4, norm_layer=norm_layer, act=act)
        self.layer2d_img = up_layer(bc * 16, bc * 8, norm_layer=norm_layer, act=act)
        self.layer3d_img = up_layer(bc * 16, bc * 16, norm_layer=norm_layer, act=act)
        self.layer4d_img = up_layer(bc * 16, bc * 16, norm_layer=norm_layer, act=act)

        self.fuse1 = nn.ModuleList([guide(bc * 4, norm_layer=norm_layer, padding_mode=padding_mode, act=act,
                                          drop_path=_img_dprs[_img_dpc + i], dilation=dilations[0][i]) for i in
                                    range(img_layers[0])])
        _img_dpc += img_layers[0]
        self.fuse2 = nn.ModuleList([guide(bc * 8, norm_layer=norm_layer, padding_mode=padding_mode, act=act,
                                          drop_path=_img_dprs[_img_dpc + i], dilation=dilations[1][i]) for i in
                                    range(img_layers[1])])
        _img_dpc += img_layers[1]
        self.fuse3 = nn.ModuleList([guide(bc * 16, norm_layer=norm_layer, padding_mode=padding_mode, act=act,
                                          drop_path=_img_dprs[_img_dpc + i], dilation=dilations[2][i]) for i in
                                    range(img_layers[2])])
        _img_dpc += img_layers[2]
        self.fuse4 = nn.ModuleList([guide(bc * 16, norm_layer=norm_layer, padding_mode=padding_mode, act=act,
                                          drop_path=_img_dprs[_img_dpc + i], dilation=dilations[3][i]) for i in
                                    range(img_layers[3])])
        _img_dpc += img_layers[3]
        self.fuse5 = nn.ModuleList([guide(bc * 16, norm_layer=norm_layer, padding_mode=padding_mode, act=act,
                                          drop_path=_img_dprs[_img_dpc + i], dilation=dilations[4][i]) for i in
                                    range(img_layers[4])])
        # _img_dpc += img_layers[4]

        self.mrf_omega = nn.Sequential(
            BasicBlock(bc * 4, bc * 4, norm_layer=nn.BatchNorm2d, padding_mode=padding_mode,
                       act=nn.GELU, last=False),
            BasicBlock(bc * 4, bc * 4, norm_layer=nn.BatchNorm2d, padding_mode=padding_mode,
                       act=nn.GELU, last=False),
            Basic2dTrans(bc * 4, 3 * pd * pg + pp * pg + 2, norm_layer=None, act=nn.Identity),

        )

        self.mrf_r = nn.Sequential(
            BasicBlock(bc * 4, bc * 4, norm_layer=nn.BatchNorm2d, padding_mode=padding_mode,
                       act=nn.GELU, last=False),
            BasicBlock(bc * 4, bc * 4, norm_layer=nn.BatchNorm2d, padding_mode=padding_mode,
                       act=nn.GELU, last=False),
            Basic2dTrans(bc * 4, 3 * pd * pg + pp * pg, norm_layer=None, act=nn.Identity),

        )

        self.damping_beta = nn.Sequential(
            BasicBlock(bc * 4, bc * 4, norm_layer=nn.BatchNorm2d, padding_mode=padding_mode,
                       act=nn.GELU, last=False),
            BasicBlock(bc * 4, bc * 4, norm_layer=nn.BatchNorm2d, padding_mode=padding_mode,
                       act=nn.GELU, last=False),
            Basic2dTrans(bc * 4, pg + pg, norm_layer=None, act=nn.Sigmoid),
        )
        self.nonlocal_offs = nn.Sequential(
            BasicBlock(bc * 4, bc * 4, norm_layer=nn.BatchNorm2d, padding_mode=padding_mode,
                       act=nn.GELU, last=False),
            BasicBlock(bc * 4, bc * 4, norm_layer=nn.BatchNorm2d, padding_mode=padding_mode,
                       act=nn.GELU, last=False),
            Basic2dTrans(bc * 4, 2 * pp * pg, norm_layer=None, act=nn.Identity),
        )

        self.prop = prop_layer(bc * 2)
        dist = torch.ones((1, 1, 1, 1), dtype=torch.float)
        self.register_buffer('dist', dist)

    def pinv(self, D, K):
        """Projects 2D pixel coordinates and depth map D into 3D point cloud (Pxyz) using camera intrinsic matrix K."""
        B, _, height, width = D.shape
        B, _, height, width = D.shape
        xx, yy = torch.meshgrid(torch.arange(width, device=D.device),
                                torch.arange(height, device=D.device),
                                indexing='xy')
        fx, fy, cx, cy = K[:, 0:1, 0:1].unsqueeze(-1), K[:, 1:2, 1:2].unsqueeze(-1), \
            K[:, 0:1, 2:3].unsqueeze(-1), K[:, 1:2, 2:3].unsqueeze(-1)
        xx = xx.view(1, 1, height, width)
        yy = yy.view(1, 1, height, width)
        Px = D * (xx - cx) / fx
        Py = D * (yy - cy) / fy
        Pz = D
        Pxyz = torch.cat([Px, Py, Pz], dim=1).contiguous()
        return Pxyz

    def forward(self, I, S, conf, eta, K, P=None, D=None, feat=[None, None, None, None, None]):
        """
        I: Bx3xHxW
        S: Bx1xHxW
        K: Bx3x3
        """
        feat_out = []
        if P is None:
            PE = self.pinv(self.dist.expand(I.shape[0], 1, *I.shape[2:]), K)
            IDP = torch.cat([I, PE], 1)
        else:
            PE = self.pinv(D, K)
            IDP = torch.cat([I, PE, torch.sigmoid(P)], 1)
        E0 = self.conv_img(IDP)
        E1 = self.layer0_img(E0)
        for fuse in self.fuse1:
            E1 = fuse(E1, feat[0])
        E2 = self.layer1_img(E1)
        for fuse in self.fuse2:
            E2 = fuse(E2, feat[1])
        E3 = self.layer2_img(E2)
        for fuse in self.fuse3:
            E3 = fuse(E3, feat[2])
        E4 = self.layer3_img(E3)
        for fuse in self.fuse4:
            E4 = fuse(E4, feat[3])
        E5 = self.layer4_img(E4)
        for fuse in self.fuse5:
            E5 = fuse(E5, feat[4])

        D5 = E5
        feat_out.append(D5)
        D4 = self.layer4d_img(D5, E4)
        feat_out.append(D4)
        D3 = self.layer3d_img(D4, E3)
        feat_out.append(D3)
        D2 = self.layer2d_img(D3, E2)
        feat_out.append(D2)
        D1 = self.layer1d_img(D2, E1)
        feat_out.append(D1)

        probability = self.mrf_omega(D1)
        diffs = self.mrf_r(D1)
        offs = self.nonlocal_offs(D1)
        betas = self.damping_beta(D1)
        eta_out, conf_out, depth, conf = self.prop([probability, diffs, offs, betas], S=S, conf_out=conf,
                                                   eta_out=eta)
        return depth, conf, eta_out, conf_out, feat_out[::-1]


class Net(nn.Module):
    """
    output conf as var
    all prior are true
    """

    def __init__(self, N=1, prop_layer=GBP, guide=FUSE, pgs=[1], pls=[1], pd=4,
                 img_layers=[2, 2, 2, 2, 2], unet=UNet, dilations=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]):
        super().__init__()
        self.N = N
        self.unets = nn.ModuleList()
        for i in range(N):
            if i:
                self.unets.append(
                    unet(inch=7, prop_layer=partial(prop_layer, pg=pgs[i], pl=pls[i]), dilations=dilations,
                         pg=pgs[i], pl=pls[i], pd=pd,
                         guide=partial(guide, fuse=True), img_layers=img_layers))
            else:
                self.unets.append(
                    unet(inch=6, prop_layer=partial(prop_layer, pg=pgs[i], pl=pls[i]), dilations=dilations,
                         pg=pgs[i], pl=pls[i], pd=pd,
                         guide=partial(guide, fuse=False), img_layers=img_layers))

    def forward(self, I, S, K):
        b, _, h, w = I.shape
        conf_out = [torch.zeros((b, 8, h, w), dtype=I.dtype, device=I.device),
                    torch.zeros((b, 1, h, w), dtype=I.dtype, device=I.device)]
        eta_out = [torch.zeros((b, 8, h, w), dtype=I.dtype, device=I.device),
                   torch.zeros((b, 1, h, w), dtype=I.dtype, device=I.device)]
        for i in range(self.N):
            if i:
                depth, conf, eta_out, conf_out, feat = self.unets[i](I=I, S=S, D=depth, P=conf, conf=conf_out, K=K,
                                                                     eta=eta_out, feat=feat)
            else:
                depth, conf, eta_out, conf_out, feat = self.unets[i](I=I, S=S, conf=conf_out, K=K, eta=eta_out)
        return depth, conf


def GBPN(batch, height, width):
    """
    """
    net = Net(N=2,
              prop_layer=functools.partial(GBP, pp=3, eps=1e-10, height=height, width=width, batch=batch),
              guide=partial(FUSE, kernel_size=7), pgs=[5, 8], pls=[4, 4], pd=4,
              dilations=[[1, 16], [1, 8], [1, 4], [1, 2], [1, 1]],
              img_layers=[2, 2, 2, 2, 2], unet=functools.partial(UNet, up_layer=UpCatIden, padding_mode='zeros'))
    net.apply(functools.partial(weights_init, mode='trunc'))
    for m in net.modules():
        if isinstance(m, FUSE):
            nn.init.constant_(m.projs.weight, 0)
            if hasattr(m, 'proj'):
                nn.init.constant_(m.proj.weight, 0)
        if isinstance(m, BasicBlock):
            nn.init.constant_(m.bn2.weight, 0)
        if isinstance(m, UNet):
            nn.init.constant_(m.mrf_r[-1].conv.weight, 0)
            nn.init.constant_(m.mrf_r[-1].conv.bias, 0)
            nn.init.constant_(m.mrf_omega[-1].conv.weight, 0)
            nn.init.constant_(m.mrf_omega[-1].conv.bias, 0)
            nn.init.constant_(m.damping_beta[-1].conv.weight, 0)
            nn.init.constant_(m.damping_beta[-1].conv.bias, 0)
            nn.init.constant_(m.nonlocal_offs[-1].conv.weight, 0)
            nn.init.constant_(m.nonlocal_offs[-1].conv.bias, 0)
    net.apply(inplace_relu)
    return net

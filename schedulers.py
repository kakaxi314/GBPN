import sys
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, OneCycleLR, LambdaLR, LinearLR, ExponentialLR


def NoiseLR(**kwargs):
    lr_sched = getattr(sys.modules[__name__], kwargs.pop('lr_sched', 'OneCycleLR'))

    class sched(lr_sched):
        def __init__(self, **kwargs):
            self.noise_pct = kwargs.pop('noise_pct', 0.1)
            self.noise_seed = kwargs.pop('noise_seed', 0)
            max_lr = []
            for para in kwargs['optimizer'].param_groups:
                if 'lr' in para:
                    max_lr.append(para['lr'])
            if len(max_lr)>0:
                kwargs['max_lr']=max_lr
            super().__init__(**kwargs)

        def get_lr(self):
            """
            lrn: Learning Rate with Noise
            """
            g = torch.Generator()
            g.manual_seed(self.noise_seed + self.last_epoch)
            noise = 2 * torch.rand(1, generator=g).item() - 1
            lrs = super().get_lr()
            lrn = []
            for lr in lrs:
                lrn.append(lr * (1 + self.noise_pct * noise))
            return lrn

    return sched(**kwargs)


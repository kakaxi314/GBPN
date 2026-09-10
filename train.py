import torch
from tqdm import tqdm
import hydra
import torch.distributed as dist
from utils import *
from omegaconf import OmegaConf


def train(run):
    run.net.train()
    for datas in tqdm(run.trainloader, desc="train", dynamic_ncols=True, leave=False, disable=run.rank):
        if run.epoch >= run.cfg.test_epoch:
            if run.iter % run.cfg.test_iter == 0:
                if run.ema:
                    test(run, ema=True, iter=True)
                else:
                    test(run, ema=False, iter=True)
        datas = run.init_cuda(*datas)
        run.net.train()
        run.optimizer.zero_grad(set_to_none=True)

        output, var = run.net(*datas[:-1])
        loss = run.criterion(output, datas[-1], var)
        loss.backward()

        if run.clip:
            grad_norm = run.clip(run.net.parameters())
            if (run.iter > 30000) and (grad_norm > OmegaConf.select(run.cfg, 'grad_norm', default=100)):
                pass
            else:
                run.optimizer.step()
        else:
            run.optimizer.step()

        if run.ema and (run.iter % run.ema_iter == 0):
            run.net_ema.update(run.net, run.epoch)
        if run.lr_iter:
            run.lr_scheduler.step()
        if run.iter % run.cfg.vis_iter == 0:
            run.writer.add_scalar("Lr", run.optimizer.param_groups[0]['lr'], run.iter)
            run.writer.add_scalar("Loss", loss.item(), run.iter)
            if run.clip:
                if grad_norm is not None:
                    run.writer.add_scalar("GradNorm", grad_norm.item(), run.iter)
        run.iter += 1
    if not run.lr_iter:
        run.lr_scheduler.step()
    run.writer.flush()


def test(run, ema=False, iter=False):
    top1 = AverageMeter()
    if ema:
        net = run.net_ema.get()
        best_metric_name = "best_metric_ema"
        legand = 'net_ema'
    else:
        net = run.net
        best_metric_name = "best_metric"
        legand = 'net'
    net.eval()
    with torch.no_grad():
        for datas in tqdm(run.testloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank):
            datas = run.init_cuda(*datas)

            output, _ = net(*datas[:-1])
            prec1 = run.metric(output, datas[-1])
            if isinstance(prec1, (list, tuple)):
                prec1 = prec1[0]
            if run.ddp:
                dist.reduce(prec1, 0, dist.ReduceOp.AVG)
            top1.update(prec1.item())
    if iter:
        run.writer.add_scalars("RMSE_Iter", {legand: top1.avg}, run.iter)
    else:
        run.writer.add_scalars("RMSE", {legand: top1.avg}, run.epoch)
    if top1.avg < getattr(run, best_metric_name):
        setattr(run, best_metric_name, top1.avg)
        run.save_state_ema(ema)
        run.ddp_cout(f'Epoch: {run.epoch} {best_metric_name}: {top1.avg:.7f}\n')

    if ema:
        del net
    # test loader change seed, re-init seed each time
    run.init_seed(iter=run.iter)

@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        for epoch in tqdm(range(run.cfg.start_epoch, run.cfg.nepoch), desc="epoch", dynamic_ncols=True):
            run.epoch = epoch
            if run.train_sampler:
                run.train_sampler.set_epoch(epoch)
            train(run)
            torch.cuda.synchronize()
            test(run, ema=False)
            torch.cuda.synchronize()
            if run.ema:
                test(run, ema=True)
            torch.cuda.synchronize()


if __name__ == '__main__':
    main()

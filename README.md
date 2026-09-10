# GBPN

## Introduction

This is the pytorch implementation of our paper **Gaussian Belief Propagation Network for Depth Completion**

📄 [Paper](https://arxiv.org/abs/2601.21291) | 🎬 [Video](https://www.youtube.com/watch?v=bijAkxdM7VY)
## Environment

We provide docker file, and you can create the image by yourself.
```
docker build -t gbpn --network=host .
```
Then you should have an image named *gbpn*.

## Dataset
We train and evaluate on KITTI and NYUV2 dataset.

### KITTI
Please download KITTI [depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
dataset.
The structure of data directory:
```
└── datas
    └── kitti
        ├── data_depth_annotated
        │   ├── train
        │   └── val
        ├── data_depth_velodyne
        │   ├── train
        │   └── val
        ├── raw
        │   ├── 2011_09_26
        │   ├── 2011_09_28
        │   ├── 2011_09_29
        │   ├── 2011_09_30
        │   └── 2011_10_03
        ├── test_depth_completion_anonymous
        │   ├── image
        │   ├── intrinsics
        │   └── velodyne_raw
        └── val_selection_cropped
            ├── groundtruth_depth
            ├── image
            ├── intrinsics
            └── velodyne_raw
```

### NYUV2
We used preprocessed NYUv2 [HDF5 dataset](http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz) provided by [Fangchang Ma](https://github.com/fangchangma/sparse-to-dense).
Note, the original full NYUv2 dataset is available at the [official website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

The structure of data directory:

```
└── datas
    └── nyudepthv2
        ├── train
        │    ├── basement_0001a
        │    │    ├── 00001.h5
        │    │    └── ...
        │    ├── basement_0001b
        │    │    ├── 00001.h5
        │    │    └── ...
        │    └── ...
        └── val
            └── official
                ├── 00001.h5
                └── ...
```



## Trained Models
You can directly download the trained models.
Our models are trained on a GPU workstation with 4 Nvidia GTX 3090.
- [GBPN_KITTI & GBPN_NYU](https://huggingface.co/PingpingXie/GBPN/tree/main)


## Start up the docker container
Run the following command, replacing the path placeholders with your local directory paths.
```
NYU_PATH=/your/nyu/path KITTI_PATH=/your/kitti/path CHPT_PATH=/your/chpt/path docker compose run gbpn
```

## Train
You can also train by yourself.

### KITTI

train on KITTI
```
torchrun --nproc_per_node=4 --master_port 3210 train.py \
gpus=[0,1,2,3] num_workers=4 name=GBPN_KITTI \
net=GBPN loss=MSAEVBYNS loss.alpha1=1e-1 loss.alpha2=1. net.clip.max_norm=1e-3 \
data=KITTI ++data.trainset.RandCrop=true \
data.mul_factor=1 ++data.trainset.tp_min=50 data.height=288 data.width=1216 \
lr=1e-3 train_batch_size=2 test_batch_size=1 \
sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
sched.lr.policy.final_div_factor=0.1  net.ema.start_epoch=3 \
net.ema.decay=0.99996 \
nepoch=30 test_epoch=25 ++tf32=true ++net.sbn=true
```

### NYUV2

train on NYUV2
```
torchrun --nproc_per_node=2 --master_port 1100 train.py \
gpus=[0,1] num_workers=4 name=GBPN_NYU \
net=GBPN loss=MSAEVBYNS loss.alpha1=1. loss.alpha2=1. net.clip.max_norm=1e-3 \
data=NYU  data.scale_factor=1.0 data.scale_dep=true \
++data.trainset.depth_var=0.4 \
data.mul_factor=1 data.num_mask=1 data.npoints=500 \
lr=2e-3 train_batch_size=8 test_batch_size=1 \
sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
sched.lr.policy.final_div_factor=0.1 net.ema.decay=0.99996 \
nepoch=100 test_epoch=80 ++tf32=true ++net.sbn=true ++data.fix_seed=true  ++grad_norm=3.0
```

## Test

With the trained model, 
you can test and save results.

### KITTI

test on KITTI selval set
```
python test.py gpus=[0] name=GBPNKITTI \
net=GBPN num_workers=4 \
train_batch_size=4  test_batch_size=1 \
data=KITTI data.height=352 data.testset.mode=selval \
metric=MetricALL ++chpt=GBPN_KITTI
```

test on KITTI test set and save for submission
```
python test.py gpus=[0] name=GBPNKITTI \
net=GBPN num_workers=4 \
train_batch_size=4  test_batch_size=1 \
data=KITTI data.height=352 data.testset.mode=test \
metric=MetricALL ++chpt=GBPN_KITTI ++save=true
```

### NYUV2

test on NYUV2 test set

```
python test.py gpus=[0] name=nyu net=GBPN num_workers=6 data=NYU \
test_batch_size=1 metric=MetricALL \
data.scale_factor=1.0 data.scale_dep=false \
data.mul_factor=1 data.num_mask=1 \
data.npoints=500 ++chpt=GBPN_NYU
```

## Citation
If you find this work useful in your research, please consider citing:
```
@article{GBPN,
title={Gaussian Belief Propagation Network for Depth Completion},
author={Tang, Jie and Xie, Pingping and Li, Jian and Tan, Ping},
journal={ECCV},
year={2026}
}
```

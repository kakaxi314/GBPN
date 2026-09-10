FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tmux && apt-get install -y libjpeg-turbo8-dev pkg-config ninja-build libopencv-dev libturbojpeg0-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir --break-system-packages \
    opencv-python==4.11.0.86  numba cupy-cuda12x pkgconfig tensorboard hydra-core einops scipy h5py timm
RUN pip install natten==0.17.5
ENV GbpnDir=/root/exts
COPY exts $GbpnDir
WORKDIR $GbpnDir
RUN rm -rf build && python setup.py install

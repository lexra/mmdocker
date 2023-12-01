ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
#FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

##########################################################
WORKDIR /root
RUN mkdir -p /docker_mount

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

##########################################################
# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y vim curl wget zip git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

##########################################################
# Install xtcocotools
RUN pip install cython xtcocotools

# Install MMPose
RUN conda clean --all

##########################################################
# MMPOSE
ENV FORCE_CUDA="1"
RUN git clone https://github.com/open-mmlab/mmpose.git && cd mmpose && git checkout v0.25.1
RUN cd mmpose && pip install -r requirements/build.txt && pip install --no-cache-dir -e .
RUN pip install mmengine
RUN pip install numpy --upgrade
RUN apt update

##########################################################
# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

##########################################################
# kneron-mmpose
RUN git clone https://github.com/kneron/kneron-mmpose.git

##########################################################
#RUN cd kneron-mmpose && mkdir -p data/freihand
#RUN wget https://download.openmmlab.com/mmpose/datasets/frei_annotations.tar -O kneron-mmpose/data/frei_annotations.tar
#RUN wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip -O kneron-mmpose/data/FreiHAND_pub_v2.zip
#RUN cd kneron-mmpose/data && tar xvf frei_annotations.tar -C freihand && unzip -o FreiHAND_pub_v2.zip -d freihand

##########################################################
#RUN https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/RHD_v1-1.zip -O kneron-mmpose/data/RHD_v1-1.zip
#RUN wget https://download.openmmlab.com/mmpose/datasets/rhd_annotations.zip -O kneron-mmpose/data/rhd_annotations.zip
#RUN cd kneron-mmpose/data && unzip -o RHD_v1-1.zip && unzip -o rhd_annotations.zip -d RHD_published_v2 && ln -sf RHD_published_v2 rhd

##########################################################
RUN sed 's|text, style_config=yapf_style, verify=True|text, style_config=yapf_style|g' -i /opt/conda/lib/python3.8/site-packages/mmcv/utils/config.py || true


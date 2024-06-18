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
ENV DEBIAN_FRONTEND=noninteractive

##########################################################
# To fix GPG key error when running apt-get update
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y vim cmake curl wget zip git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global http.postBuffer 524288000
RUN git config --global core.compression 0

##########################################################
# Install xtcocotools
RUN pip install pip --upgrade
RUN pip install cython xtcocotools
RUN pip install gdown==5.1.0

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
RUN sed 's|text, style_config=yapf_style, verify=True|text, style_config=yapf_style|g' -i /opt/conda/lib/python3.8/site-packages/mmcv/utils/config.py || true

##########################################################
# kneron-mmpose
RUN git clone https://github.com/kneron/kneron-mmpose.git

##########################################################
RUN cd kneron-mmpose && mkdir -p data/freihand
RUN wget https://download.openmmlab.com/mmpose/datasets/frei_annotations.tar -O kneron-mmpose/data/frei_annotations.tar
RUN wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip -O kneron-mmpose/data/FreiHAND_pub_v2.zip
RUN cd kneron-mmpose/data && tar xvf frei_annotations.tar -C freihand && unzip -o FreiHAND_pub_v2.zip -d freihand

##########################################################
#RUN https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/RHD_v1-1.zip -O kneron-mmpose/data/RHD_v1-1.zip
#RUN wget https://download.openmmlab.com/mmpose/datasets/rhd_annotations.zip -O kneron-mmpose/data/rhd_annotations.zip
#RUN cd kneron-mmpose/data && unzip -o RHD_v1-1.zip && unzip -o rhd_annotations.zip -d RHD_published_v2 && ln -sf RHD_published_v2 rhd

##########################################################
#RUN git clone https://github.com/ARM-software/ML-zoo.git

##########################################################
# tesseract
RUN git clone https://github.com/tesseract-ocr/tesseract.git
RUN cd tesseract && git checkout 4.0
RUN git clone https://github.com/tesseract-ocr/langdata_lstm.git
RUN apt install -y tesseract-ocr libtesseract-dev libicu-dev libpango1.0-dev libcairo2-dev libleptonica-dev
RUN apt install -y fonts-ubuntu fonts-dejavu-core fonts-dejavu-extra fonts-arphic-uming fonts-cwtex-ming \
	fonts-droid-fallback fonts-freefont-ttf fonts-cwtex-ming fonts-cwtex-yen
#RUN cd tesseract && src/training/tesstrain.sh --fonts_dir /usr/share/fonts --lang eng --linedata_only --noextract_font_properties --langdata_dir ../langdata_lstm --tessdata_dir ./tessdata --output_dir ~/tesstutorial/engtrain --fontlist 'FreeMono' 'DejaVu Serif' 'ubuntu' 'AR PL UMing TW Light' 'cwTeXMing Medium' 'cwTeXYen Medium' 'Droid Sans Fallback Regular' --overwrite

##########################################################
# deep-text-recognition-benchmark
RUN pip install trdg
RUN git clone https://github.com/Belval/TextRecognitionDataGenerator.git
RUN cd TextRecognitionDataGenerator && pip install --no-cache-dir -e .
RUN pip install --ignore-installed Pillow==9.3.0
#RUN git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
RUN git clone https://github.com/zihaomu/deep-text-recognition-benchmark.git
RUN pip install lmdb natsort nltk fire
RUN cd deep-text-recognition-benchmark && ln -sf ../../docker_mount/data_lmdb_release .
COPY TPS-ResNet-BiLSTM-Attn.pth deep-text-recognition-benchmark
#RUN git clone https://github.com/parksunwoo/ocr_kor.git
#RUN git clone https://github.com/JaidedAI/EasyOCR.git

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install numpy==1.19.5
RUN pip install scikit-learn --upgrade
#RUN cd deep-text-recognition-benchmark && CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC

##########################################################
# crnn.pytorch
RUN pip install numpy==1.23.5
RUN pip install protobuf==3.20.3
RUN pip install TensorRT==8.5.3.1
RUN pip install onnxruntime==1.5.2
RUN pip install onnx==1.7.0
RUN pip install pydantic==1.7
RUN git clone https://github.com/YIYANGCAI/CRNN-Pytorch2TensorRT-via-ONNX.git
#RUN git clone https://github.com/meijieru/crnn.pytorch
RUN git clone https://github.com/Holmeyoung/crnn-pytorch.git
RUN cd crnn-pytorch && gdown https://drive.google.com/uc?id=1FQgOoqvUvzSGPbxRVL1Bgvl-0AUeHQ8N

##########################################################
# crnn
#RUN git clone https://github.com/SeanNaren/warp-ctc.git && cd warp-ctc && mkdir build && cd build
#RUN cd warp-ctc && git checkout CMakeLists.txt && sed '35d' -i CMakeLists.txt
#RUN mkdir -p warp-ctc/build && cd warp-ctc/build && cmake .. && make -j8
#RUN cd warp-ctc/pytorch_binding && python setup.py install && cp ../build/libwarpctc.so /opt/conda/lib/
#RUN pip install Levenshtein
#RUN pip install tensorflow==2.13.1 scipy==1.5.0 matplotlib==3.4.3 xtcocotools==1.8 torchsummary onnx-tf==1.9.0
#RUN git clone https://github.com/pbcquoc/crnn.git
##RUN cd crnn && sed 's|text = labels\[i\]|text = labels\[0\]|' -i models/utils.py

##########################################################
# HybridNets
RUN git clone https://github.com/datvuthanh/HybridNets.git
COPY HybridNets/export-hybridnets.py HybridNets/export.py
COPY HybridNets/requirements-hybridnets.txt HybridNets/requirements.txt
RUN cd HybridNets && pip install -r requirements.txt \
	&& curl --create-dirs -L -o weights/hybridnets.pth https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth
RUN pip3 uninstall -y albucore
RUN pip install numpy==1.23.5
RUN cd HybridNets && python export.py && rm -rfv weights/hybridnets.pt
#RUN cd HybridNets && python hybridnets_test.py -w weights/hybridnets.pth --source demo/image --output demo_result --imshow False --imwrite True --float16 True

##########################################################
# Yolo-Fastest
RUN git clone https://github.com/HimaxWiseEyePlus/Yolo-Fastest.git && sed 's|^CUDNN_HALF=0|CUDNN_HALF=1|g' -i Yolo-Fastest/Makefile
RUN cd Yolo-Fastest && git clone https://github.com/HimaxWiseEyePlus/keras-YOLOv3-model-set.git
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib && export PATH=/usr/local/cuda/bin:$PATH && make -C Yolo-Fastest

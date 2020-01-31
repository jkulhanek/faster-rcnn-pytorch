FROM ubuntu:latest

# Create a working directory
RUN mkdir /app && mkdir /datasets
WORKDIR /app

ENV GOOGLE_CLOUD_SDK_VERSION 203.0.0

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    gcc \
    apt-transport-https \
    lsb-release \
    openssh-client \
    libx11-6 \
    gnupg \
    make \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/* \
# Create a non-root user and switch to it
 && adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app  && chown -R user:user /datasets \
 && echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user && \
# Install Miniconda
curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh && \
export PATH=/home/user/miniconda/bin:$PATH && \
 #
 # Create a Python 3.6 environment
 /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.4 \
 && /home/user/miniconda/bin/conda clean -ya && \
 export CONDA_DEFAULT_ENV=py36 && \
 export CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV && \
 export PATH=$CONDA_PREFIX/bin:$PATH && \
 #
 # Ensure conda version is at least 4.4.11
 # (because of this issue: https://github.com/conda/conda/issues/6811)
 export CONDA_AUTO_UPDATE_CONDA=false && \
 conda install -y "conda>=4.4.11" && conda clean -ya && \
 # No CUDA-specific steps
 export NO_CUDA=1 && \
 conda install -y -c pytorch \
    pytorch=0.4.0 \
    torchvision=0.2.1 \
 && conda clean -ya \
# Install HDF5 Python bindings
 && conda install -y \
    h5py \
 && conda clean -ya \
 && pip install h5py-cache \
# Install Torchnet, a high-level framework for PyTorch
 && pip install git+https://github.com/pytorch/tnt.git@master \
# Install Requests, a Python library for making HTTP requests
 && conda install -y requests && conda clean -ya \
# Install Graphviz
 && conda install -y graphviz=2.38.0 \
 && conda clean -ya \
 && pip install graphviz \
# Install OpenCV3 Python bindings
 && sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apdockt/lists/* \
 && conda install -y -c menpo opencv3 \
 && conda clean -ya \
 # Install matplotlib
 && pip install PyQt5 && \
 conda install -y matplotlib && \
# install pycoco
 conda install Cython h5py -y && conda install -y gcc_linux-64 gxx_linux-64 matplotlib \
 && conda clean -ya \
 # we have to clone the new repo and run build manually
 && /bin/bash -c "source activate root && cd /tmp && git clone https://github.com/cocodataset/cocoapi.git && cd cocoapi/PythonAPI && make install && cd /app && rm -rf /tmp/cocoapi" 
 # && \
 # INstall gcloud
# export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
# echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - && \
# sudo apt-get update && sudo apt-get install -y google-cloud-sdk && \
# gcloud config set core/disable_usage_reporting true && \
# gcloud config set component_manager/disable_update_check true && \
# gcloud config set metrics/environment github_docker_image
 # Attach the actual dataset

# This will be attached from the host
#RUN mkdir /datasets/coco && mkdir /datasets/coco/val2017 && mkdir /datasets/coco/val2017/images && mkdir /datasets/coco/val2017/annotations && \
# gsutil -m rsync gs://images.cocodataset.org/val2017 /datasets/coco/val2017/images/ && gsutil -m rsync gs://images.cocodataset.org/annotations /datasets/coco/val2017/annotations/

ENV PATH=/home/user/miniconda/envs/py36/bin:$PATH
ENV NO_CUDA=1

# Predownload VGG16
RUN python -c "from torchvision.models.vgg import model_urls;import torch.utils.model_zoo as model_zoo;model_zoo.load_url(model_urls['vgg16'])"

COPY src /app
 

# Set the default command to python3
CMD ["python3"]
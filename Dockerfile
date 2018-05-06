#FROM debian:stretch
FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# install debian packages
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
  build-essential \ 
  # install python 3
  python3.5 \ 
  python3-dev \
  python3-pip \ 
  python3-wheel \ 
  # Boost for dlib
  cmake \
  libboost-all-dev \ 
  # requirements for keras
  python3-h5py \
  python3-yaml \
  python3-pydot \
  python3-setuptools \
  ffmpeg \
  # For Docker (https://docs.docker.com/install/linux/docker-ce/debian/#set-up-the-repository)
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg2 \
  software-properties-common

#RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
#RUN add-apt-repository \
#  "deb [arch=amd64] https://download.docker.com/linux/debian stretch stable" \
# && apt-get update \
# && apt-get install -y docker-ce

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - 
RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu xenial stable" \
 && apt-get update \
 && apt-get install -y docker-ce

#RUN add-apt-repository \
  #"deb http://httpredir.debian.org/debian/ stretch main contrib non-free" \
 #&& apt-get update \
 #&& apt-get install -y linux-headers-amd64 nvidia-driver nvidia-cuda-toolkit

# Install Nvidia Docker
RUN curl -fsSLO https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
RUN dpkg -i nvidia-docker*.deb

COPY ./requirements.txt .
RUN pip3 --no-cache-dir install -r ./requirements.txt

#RUN apt-get clean \
# && rm -rf /var/lib/apt/lists/*
WORKDIR /srv/


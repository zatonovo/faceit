#FROM debian:stretch
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

# install debian packages
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
  # install essentials
  build-essential vim less\ 
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
  imagemagick libmagick++-dev \
  # For Docker (https://docs.docker.com/install/linux/docker-ce/debian/#set-up-the-repository)
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg2 \
  software-properties-common

RUN add-apt-repository -y ppa:jonathonf/ffmpeg-3 \
 && apt-get update \
 && apt-get install --no-install-recommends -y \
  ffmpeg

COPY ./requirements.txt .
RUN pip3 --no-cache-dir install -r ./requirements.txt

# These need to be installed in the image as well.
# They are already downloaded by the Makefile
COPY ./libcudnn6_6.0.21-1+cuda8.0_amd64.deb .
COPY ./libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb .
RUN dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
RUN dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb

RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

#RUN apt-get clean \
# && rm -rf /var/lib/apt/lists/*
WORKDIR /srv/


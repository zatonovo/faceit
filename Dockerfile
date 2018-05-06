#FROM debian:stretch
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

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

COPY ./requirements.txt .
RUN pip3 --no-cache-dir install -r ./requirements.txt

#RUN apt-get clean \
# && rm -rf /var/lib/apt/lists/*
WORKDIR /srv/


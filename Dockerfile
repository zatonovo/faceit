FROM debian:stretch

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
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN wget -O - -q 'https://gist.githubusercontent.com/allenday/f426e0f146d86bfc3dada06eda55e123/raw/41b6d3bc8ab2dfe1e1d09135851c8f11b8dc8db3/install-cuda.sh' | sudo bash
RUN wget -O - -q 'https://gist.githubusercontent.com/allenday/c875eaf21a2b416f6478c0a48e428f6a/raw/f7feca1acc1a992afa84f347394fd7e4bfac2599/install-docker-ce.sh' | sudo bash
wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i nvidia-docker*.deb

COPY ./requirements.txt .
RUN pip3 --no-cache-dir install -r ./requirements.txt

WORKDIR /srv/


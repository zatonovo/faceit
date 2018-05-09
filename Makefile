# Install on Ubuntu 16.04 (xenial) or Debian 9 (stretch)
# Requirements
# apt-get install -y git curl wget (provided by default image in GCP)
# apt-get install -y build-essential
#
# References
# https://yangcha.github.io/Install-CUDA8/
# https://medium.com/google-cloud/jupyter-tensorflow-nvidia-gpu-docker-google-compute-engine-4a146f085f17
# https://askubuntu.com/questions/873112/imagemagick-cannot-be-detected-by-moviepy
#
# Check if GPUs are visible from a docker container
# sudo nvidia-docker-plugin
#
# Verify an actual container can see GPUs
# sudo nvidia-docker run --rm nvidia/cuda nvidia-smi
#
# Launch Tensorboard
# sudo nvidia-docker run --rm --name tf1 -p 8888:8888 -p 6006:6006 gcr.io/tensorflow/tensorflow:latest-gpu jupyter notebook --allow-root
#
# The build for local work
# make all
#
# The build for GPU machine
# make init-ubuntu
# make all
# make demo

# Set to cpu for cpu build: make ARCH=cpu all
ARCH ?= gpu

all: get-cuda
	mkdir -p data/persons
	mkdir -p data/output
	cp requirements-${ARCH}.txt requirements.txt
	docker build -t deepfakes .

get-cuda:
	test -f "cuda-repo-ubuntu1604_8.0.61-1_amd64.deb" || wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	test -f "libcudnn6_6.0.21-1+cuda8.0_amd64.deb" || wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6_6.0.21-1%2Bcuda8.0_amd64.deb
	test -f "libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb" || wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6-dev_6.0.21-1%2Bcuda8.0_amd64.deb

init:
	git submodule update --init --recursive
	dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
	dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
	apt-get update
	apt-get install -y cuda=8.0.61-1
	apt-get install -y libcudnn6-dev
	echo "Please comment out the line <policy domain="path" rights="none" pattern="@*" /> in /etc/ImageMagick-6/policy.xml"


init-ubuntu: init
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - 
	add-apt-repository \
	   "deb [arch=amd64] https://download.docker.com/linux/ubuntu xenial stable" \
	 && apt-get update \
	 && apt-get install -y docker-ce
	# Install Nvidia Docker
	curl -fsSLO https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
	dpkg -i nvidia-docker*.deb

init-debian: init
	curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
	add-apt-repository \
	  "deb [arch=amd64] https://download.docker.com/linux/debian stretch stable" \
	 && apt-get update \
	 && apt-get install -y docker-ce
	
	add-apt-repository \
	  "deb http://httpredir.debian.org/debian/ stretch main contrib non-free" \
	 && apt-get update \
	 && apt-get install -y linux-headers-amd64 nvidia-driver nvidia-cuda-toolkit
	# Install Nvidia Docker
	curl -fsSLO https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
	dpkg -i nvidia-docker*.deb



run-cpu:
	docker run -p 8888:8888 -p 6006:6006 -v $(shell pwd):/srv -it --rm deepfakes

run:
	nvidia-docker run -p 8888:8888 -p 6006:6006 -v $(shell pwd):/srv -it --rm deepfakes

demo: run
	echo "Get example faces"
	curl https://gazettereview.com/wp-content/uploads/2017/04/jimmy-fallon.jpg -o data/persons/fallon.jpg
	curl http://i.huffpost.com/gen/2571402/thumbs/o-JOHN-OLIVER-facebook.jpg -o data/persons/oliver.jpg
	echo "Download videos and extract faces"
	echo "python3 faceit.py preprocess fallon_to_oliver"
	echo "Training model"
	echo "python3 faceit.py train fallon_to_oliver"
	echo "python3 faceit.py convert fallon_to_oliver fallon_emmastone.mp4 --start 40 --duration 55 --side-by-side"


# Install on Ubuntu 16.04 (xenial) or Debian 9 (stretch)
# Check Nvidia/Docker installation by running
# sudo nvidia-docker-plugin # Checks if GPUs are visible from a docker container
# sudo nvidia-docker run --rm nvidia/cuda nvidia-smi # Verifies an actual container can see GPUs
#
# Launch Tensorboard
# sudo nvidia-docker run --rm --name tf1 -p 8888:8888 -p 6006:6006 gcr.io/tensorflow/tensorflow:latest-gpu jupyter notebook --allow-root


all:
	mkdir -p data/persons
	docker build -t deepfakes .

init:
	git submodule update --init --recursive
	wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6_6.0.21-1%2Bcuda8.0_amd64.deb
	wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6-dev_6.0.21-1%2Bcuda8.0_amd64.deb
	dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
	dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
	apt-get update
	apt-get install cuda=8.0.61-1
	apt-get install libcudnn6-dev
	# Install Nvidia Docker
	curl -fsSLO https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
	dpkg -i nvidia-docker*.deb

init-ubuntu: init
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - 
	add-apt-repository \
	   "deb [arch=amd64] https://download.docker.com/linux/ubuntu xenial stable" \
	 && apt-get update \
	 && apt-get install -y docker-ce

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



run:
	nvidia-docker run --name deepfakes -p 8888:8888 -p 6006:6006 -v $(shell pwd):/srv -it deepfakes

demo: run
	echo "Get example faces"
	wget https://gazettereview.com/wp-content/uploads/2017/04/jimmy-fallon.jpg -O data/persons/fallon.jpg
	wget http://i.huffpost.com/gen/2571402/thumbs/o-JOHN-OLIVER-facebook.jpg -O data/persons/oliver.jpg
	echo "Download videos and extract faces"
	echo "python3 faceit.py preprocess fallon_to_oliver"
	echo "Training model"
	echo "python3 faceit.py train fallon_to_oliver"

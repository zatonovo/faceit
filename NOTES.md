https://github.com/goberoi/faceit
https://github.com/deepfakes/faceswap
https://github.com/shaoanlu/faceswap-GAN

[Prerequisites](https://medium.com/google-cloud/jupyter-tensorflow-nvidia-gpu-docker-google-compute-engine-4a146f085f17)

Set up [firewall rules](https://console.cloud.google.com/networking/firewalls/list) on GCP.
Need one for Jupyter (8888) and TensorBoard (6006).

allow-jupyter, 0.0.0.0/0, tcp:8888
allow-tensorboard, 0.0.0.0/0, tcp:6006


```{bash}
wget -O - -q 'https://gist.githubusercontent.com/allenday/f426e0f146d86bfc3dada06eda55e123/raw/41b6d3bc8ab2dfe1e1d09135851c8f11b8dc8db3/install-cuda.sh' | sudo bash

# Verify CUDA installation successful
nvidia-smi

wget -O - -q 'https://gist.githubusercontent.com/allenday/c875eaf21a2b416f6478c0a48e428f6a/raw/f7feca1acc1a992afa84f347394fd7e4bfac2599/install-docker-ce.sh' | sudo bash
wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i nvidia-docker*.deb

# Verify GPU is visible from container
sudo nvidia-docker-plugin &
sudo nvidia-docker run --rm nvidia/cuda nvidia-smi
```
Create snapshot


```{bash}
git clone https://github.com/goberoi/faceit.git
cd faceit
git submodule update --init --recursive

cd faceswap
# GPU
sudo docker build -t deepfakes-gpu -f Dockerfile.gpu .

# CPU
sudo docker build -t deepfakes-cpu .

cd ..
```
Create snapshot


```{bash}
mkdir -p data/persons

# GPU
nvidia-docker run --name deepfakes-gpu -p 8888:8888 -v [src_folder]:/srv -it deepfakes-gpu
# CPU
docker run --name deepfakes-gpu -p 8888:8888 -v $(pwd):/srv -it deepfakes-cpu

#docker exec -it deepfakes bash
```

From within container
```{bash}
x apt-get update
x apt-get install ffmpeg
pip3 install -r requirements.txt
pip3 install requests

wget https://gazettereview.com/wp-content/uploads/2017/04/jimmy-fallon.jpg -O data/persons/fallon.jpg
wget http://i.huffpost.com/gen/2571402/thumbs/o-JOHN-OLIVER-facebook.jpg -O data/persons/oliver.jpg

echo "Download videos and extract faces"
python3 faceit.py preprocess fallon_to_oliver

echo "Training model"
python3 faceit.py train fallon_to_oliver

echo "Convert video"
python faceit.py convert fallon_to_oliver fallon_emmastone.mp4 --start 40 --duration 55 --side-by-side
```


## Data Sources
Fakes:
https://deepfakeshub.net/category/gillian-anderson/

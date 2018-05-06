
all:
	mkdir -p data/persons
	docker build -t deepfakes .

init:
	git submodule update --init --recursive

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

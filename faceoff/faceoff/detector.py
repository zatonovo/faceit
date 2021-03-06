# During fine-tuning, we freeze the first 36 layers which corresponds to the first 4 blocks of the network. Only the last layer is replaced by a dense layer with two outputs, initialized randomly and trained anew for 10 epochs. After that, we train the resulting network until the validation does not change in 5 consecutive epochs. For optimization, we use the following for our reported scores: ADAM [58] with a learning rate of 0.001, β1 = 0.9 and β2 = 0.999 as well as a batch-size of 64.
# https://github.com/keras-team/keras/blob/master/keras/applications/xception.py# https://medium.com/google-cloud/serverless-transfer-learning-with-cloud-ml-engine-and-keras-335435f31e15

#from keras.applications.xception import Xception, preprocess_input
#model = Xception(weights='imagenet')
"""

Extract frames  from fake videos. This can be done locally to save hosting fees.
$ sudo make ARCH=cpu run

from faceoff import video

face_example = 'data/persons/oliver.jpg'
frame_path = 'data/processed/fallon_emmastone_fake.mp4_frames'
face_path = 'data/processed/fallon_emmastone_fake.mp4_faces'

# Extract frames from original/swapped stacked video
v = video.read_video_file('fallon_emmastone.mp4')
video.extract_frames(v, frame_path, video.crop_frame)


This needs to be done using GPU.
$ sudo tar cf fallon_emmastone_fake.mp4_frames.tar fallon_emmastone_fake.mp4_frames
$ gcloud compute --project "panoptez" scp --zone "us-central1-f" fallon_emmastone_fake.mp4_frames.tar brian@deep-learning-1:workspace/faceit/data/processed/fallon_emmastone_fake.mp4_frames.tar
$ gcloud compute --project panoptez ssh --zone us-central1-f deep-learning-1

On remote, untar the images
$ cd workspace/faceit/data/processed
$ tar xf fallon_emmastone_fake.mp4_frames.tar
$ cd ~/workspace/faceit
$ sudo make run

# Extract faces from fake video frames
from faceoff import video
face_example = 'data/persons/oliver.jpg'
frame_path = 'data/processed/fallon_dakota_fake.mp4_frames'
face_path = 'data/processed/fallon_dakota_fake.mp4_faces'

# Skip this if you did it locally
v = video.read_video_file('data/output/fallon_dakota.mp4')
video.extract_frames(v, frame_path)

video.extract_faces(frame_path, face_path, face_example, processes=1)

# Now classify fakes from reals
from faceoff import detector

(X,y) = detector.create_dataset(
  ['data/processed/fallon_emmastone.mp4_faces'], 
  ['data/processed/fallon_emmastone_fake.mp4_faces'])
data = detector.split_data(X,y)

model = detector.create_model()
model = detector.train_detector(model, data)

0.9913809/3809 [==============================] - 16s 4ms/step - loss: 0.0294 - acc: 0.9913 - val_loss: 0.5343 - val_acc: 0.8122
953/953 [==============================] - 3s 3ms/step
Loss 0.5342530553763711, Accuracy 0.8121720884554283


del X, y, data

# Validation set
from faceoff import video, detector
face_example = 'data/persons/oliver.jpg'
frame_path = 'data/processed/fallon_dakota_fake.mp4_frames'
face_path = 'data/processed/fallon_dakota_fake.mp4_faces'

# NOTE: Frames should already exist from video generation
v = video.read_video_file('data/output/fallon_dakota.mp4')
video.extract_frames(v, frame_path)
video.extract_faces(frame_path, face_path, face_example, processes=1)

(X_val,y_val) = detector.create_dataset(
  ['data/processed/oliver_pastor.mp4_faces'], 
  ['data/processed/fallon_dakota_fake.mp4_faces'])

detector.validate_model(model, X_val, y_val)


5045/5045 [==============================] - 17s 3ms/step
Loss 2.80179295832858, Accuracy 0.2836471754802821
(2.80179295832858, 0.2836471754802821)
"""

import numpy as np
import sklearn
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions

# The dataset is compressed as NumPy format 
#import requests
#url = 'https://github.com/hayatoy/deep-learning-datasets/releases/download/v0.1/tl_opera_capitol.npz'
#response = requests.get(url)
#dataset = np.load(BytesIO(response.content))
#
#X_dataset = dataset['features']
#y_dataset = dataset['labels']

# Make input data from Jpeg file
#img_path = 'seagull.jpg'
#img = image.load_img(img_path, target_size=(299, 299))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#
# Compare with
# cv2.imread('data/processed/fallon_emmastone_fake.mp4_faces/frame_7240.jpg')
# which uses BGR color-space





def create_model():
  """
  @example
  model = detector.create_model()
  detector.train_detector(model, data)
  """
  base = InceptionV3(weights='imagenet')

  # Extract intermediate layer outputs
  # The model which outputs intermediate layer features
  intermediate_layer_model = Model(
    inputs=base.input, outputs=base.layers[311].output)

  # Connect Dense layers at the end for fine tuning
  x = intermediate_layer_model.output
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(2, activation='softmax')(x)

  # Transfer Learning model
  transfer_model = Model(
    inputs=intermediate_layer_model.input, outputs=predictions)

  # Freeze all layers
  for layer in transfer_model.layers: layer.trainable = False

  # Unfreeze last dense layers
  transfer_model.layers[312].trainable = True
  transfer_model.layers[313].trainable = True

  transfer_model.compile(loss='categorical_crossentropy',
    optimizer='adam', metrics=['accuracy'])

  return transfer_model


def train_detector(model, data, epochs=20):
  """
  model = create_model()
  data = split_data()
  model = train_detector(model, data)
  """
  (X_train,X_test, y_train,y_test) = data
  model.fit(X_train, y_train, epochs=epochs,
    validation_data=(X_test, y_test))
  loss, acc = model.evaluate(X_test, y_test)
  print('Loss {}, Accuracy {}'.format(loss, acc))
  return model

def validate_model(model, X_val, y_val):
  X_val = preprocess_input(X_val)
  y_val = np_utils.to_categorical(y_val)
  loss, acc = model.evaluate(X_val, y_val)
  print('Loss {}, Accuracy {}'.format(loss, acc))
  return (loss, acc)


def is_fake(frames, model):
  preds = model.predict(frames)
  print('Predicted:')
  for p in decode_predictions(preds, top=5)[0]:
    print("Score {}, Label {}".format(p[2], p[1]))
  return preds


def create_dataset(real_dirs, fake_dirs):
  """
  @param real List of paths to folders of real faces
  @param fake List of paths to folders of fake faces
  
  @example 
  from faceoff import detector

  (X,y) = detector.create_dataset(
    ['data/processed/fallon_emmastone.mp4_faces'], 
    ['data/processed/fallon_emmastone_fake.mp4_faces'])
  data = detector.split_data(X,y)
  """
  import itertools, glob

  def read_image(path):
    x = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    x = np.expand_dims(x, axis=0)
    return x

  def read_folder(path):
    p = "{}/*.jpg".format(path)
    print("Read images from {}".format(p))
    xs = [ read_image(f) for f in glob.glob(p) ]
    print("Loaded {} images".format(len(xs)))
    return xs

  def read_folders(paths):
    xs = [ read_folder(x) for x in paths ]
    xs = list(itertools.chain.from_iterable(xs))
    return np.concatenate(xs)

  reals = read_folders(real_dirs)
  fakes = read_folders(fake_dirs)
  features = np.concatenate([reals, fakes]).astype('float32')
  labels = np.concatenate([np.zeros(reals.shape[0]), np.ones(fakes.shape[0])])

  return (features, labels)


def split_data(X, y):
  from keras.utils import np_utils
  from sklearn.model_selection import train_test_split

  X = preprocess_input(X)
  y = np_utils.to_categorical(y)
  return train_test_split(X, y, test_size=0.2, random_state=42)


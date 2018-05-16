'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

def chunk_text(text, maxlen=40, step=3, n=None):
  """
  @return (chunk, next char)
  @example
  (ci,ic) = char_gan.get_char_map(text)
  x = char_gan.to_one_hot(char_gan.chunk_text(text, n=2)[0], ci, 40)

  o = char_gan.train_gan(text, only_discriminator=True)
  o['discriminator'].predict(x[0])
  """
  if n is None:
    length = len(text)
  else:
    length = min(len(text), n * step + maxlen)
  it = range(0, length - maxlen, step)
  return list(zip(*[ (text[i:i+maxlen], text[i+maxlen]) for i in it ]))


def sample_generator(model, n, indices_char, maxlen, temp=1.0):
  """
  sample_generator(generator, 100, indices_char, 40)
  """
  def sample_one(x):
    next_index = sample(x, temp)
    next_char = indices_char[next_index]
    return next_char

  def sample_chunk(i):
    return ''.join([ sample_one(preds[i,j,:]) for j in range(maxlen) ])
  
  #x = np.random.randint(2, size=(n, maxlen, len(indices_char)))
  x = get_random_one_hot(n, maxlen, len(indices_char))
  preds = model.predict(x, verbose=0)
  preds.shape = (n,maxlen, len(indices_char))
  return ''.join([ sample_chunk(i) for i in range(n) ])


def permute_word(text, n=1):
  """
  Create a random permutation of words in text.
  """
  x = text.split(' ')
  return ' '.join(np.random.permutation(x))


def permute_text(text, n=1):
  """
  Create a random permutation of text.
  """
  return ''.join(np.random.permutation(list(text)))


def get_training_set_d(text, model, indices_char, maxlen, rn=4e4, pn=1e4):
  """
  Construct training set for discriminator. Uses 3 sources:
  1. original text (real)
  2. random crap from untrained generator (fake)
  3. random word permutations of original text (fake)
  4. random text permutations of original text (fake)

  1 is real, 0 is fake
  """
  print("Construct discriminator training set")
  (original,_) = chunk_text(text)
  (rand_text,_) = chunk_text(sample_generator(model, int(rn), indices_char, maxlen))
  (perm_word,_) = chunk_text(permute_word(text, int(pn)))
  (perm_text,_) = chunk_text(permute_text(text, int(pn)))
  fake = rand_text + perm_word + perm_text
  ys = np.concatenate([
    np.ones(len(original), dtype=np.int),
    np.zeros(len(fake), dtype=np.int)])
  msg = "Using {} original, {} random, {} permuted word, {} permuted text"
  print(msg.format(len(original), len(rand_text),len(perm_word),len(perm_text)))
  return (original + fake, ys)
    

def to_one_hot(sentences, char_indices, maxlen, next_chars=None):
  """
  Convert sentences and next character in sequence into one hot arrays
  """
  print('Convert to one hot...')
  x = np.zeros((len(sentences), maxlen, len(char_indices)), dtype=np.bool)
  y = np.zeros((len(sentences), len(char_indices)), dtype=np.bool)
  for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    if next_chars is not None:
      y[i, char_indices[next_chars[i]]] = 1
  return (x,y)


def get_char_map(text):
  chars = sorted(list(set(text)))
  print('total chars:', len(chars))
  char_indices = dict((c, i) for i, c in enumerate(chars))
  indices_char = dict((i, c) for i, c in enumerate(chars))
  return (char_indices, indices_char)


def train_discriminator(text, model_d, generator, maxlen, epochs=1):
  (char_indices, indices_char) = get_char_map(text)
  (x,y) = get_training_set_d(text, generator, indices_char, maxlen)
  #import pdb;pdb.set_trace()
  (X,_) = to_one_hot(x, char_indices, maxlen)
  #print("Fit model")
  #discriminator.summary()
  loss_d = model_d.fit(X,y, batch_size=128, epochs=epochs)
  return loss_d

def get_discriminator(maxlen, charlen):
  """
  @param maxlen Character window size
  @param charlen Number of unique characters in lexicon
  """
  print("Build discriminator")
  discriminator = Sequential()
  discriminator.add(LSTM(128, input_shape=(maxlen, charlen)))
  #discriminator.add(Flatten())
  discriminator.add(Reshape((128,)))
  discriminator.add(Dense(1, activation='sigmoid'))

  model_d = Sequential()
  model_d.add(discriminator)
  optimizer_d = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
  model_d.compile(loss='binary_crossentropy', optimizer=optimizer_d,
    metrics=['accuracy'])

  return (discriminator, model_d)


def get_generator(maxlen, charlen):
  """
  Input are random one hot vectors
  Output layer is charlen probabilities
  @param maxlen Character window size
  @param charlen Number of unique characters in lexicon
  """
  print("Build generator")
  n = maxlen * charlen
  generator = Sequential()
  generator.add(LSTM(64, input_shape=(maxlen, charlen), return_sequences=True))
  generator.add(Dropout(.2))
  generator.add(LSTM(64))
  generator.add(Reshape((64,)))
  generator.add(Dense(n))
  generator.add(Activation('softmax'))
  return generator


def get_adversary(generator, discriminator):
  print("Build adversarial model")
  model_a = Sequential()
  model_a.add(generator)
  model_a.add(Reshape(discriminator.input_shape[1:]))
  model_a.add(discriminator)
  #optimizer_a = RMSprop(lr=0.01)
  #model_a.compile(loss='categorical_crossentropy', optimizer=optimizer_a)
  #optimizer_a = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
  optimizer_a = RMSprop(lr=0.0004)
  model_a.compile(loss='binary_crossentropy', optimizer=optimizer_a,
    metrics=['accuracy'])
  return model_a


def sample(preds, temperature=1.0):
  # helper function to sample an index from a probability array
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)


def get_random_one_hot(n, maxlen, charlen):
  c = np.random.randint(charlen, size=n*maxlen)
  X = np.zeros((n, maxlen, charlen))
  for i in range(n):
    for j in range(maxlen):
      k = i*maxlen + j
      X[i,j, c[k]] = 1
  return X


def on_epoch_end(model, ic, maxlen, n=20):
  def f(epoch, logs):
    """
    Function invoked at end of each epoch. Prints generated text.
    """
    print()
    ds = [.5] #[0.2, 0.5, 1.0, 1.2]
    for diversity in ds:
      print('[epoch {}] diversity: {}'.format(epoch,diversity))
      s = sample_generator(model, n, ic, maxlen, diversity)
      print('[epoch {}] text: {}'.format(epoch,s))
    print()
  return f


def train_gan(text, maxlen=40, step=3, epochs_d=5, epochs_a=60,
    model_d=None, only_discriminator=False):
  """
  text = char_gan.read_text()
  out = char_gan.train_gan(text)
  """
  (char_indices, indices_char) = get_char_map(text)

  # cut the text in semi-redundant sequences of maxlen characters
  (sentences, next_chars) = chunk_text(text, maxlen, step)
  print('nb sequences:', len(sentences))

  generator = get_generator(maxlen, len(char_indices))
  if model_d is None:
    (discriminator, model_d) = get_discriminator(maxlen, len(char_indices))
    history_d = train_discriminator(text,
      model_d, generator, maxlen, epochs=epochs_d)
    print("Freeze discriminator")
    for layer in discriminator.layers: layer.trainable = False
  else:
    print("Use supplied discriminator")
    discriminator = model_d.layers[0]
    history_d = None

  if only_discriminator: 
    return { 'discriminator': model_d, 'history_d': history_d }

  model_a = get_adversary(generator, discriminator)

  print_callback = LambdaCallback(
    on_epoch_end=on_epoch_end(generator, indices_char, maxlen))

  n = int((len(text) - maxlen) / 3)
  print("Training on {} noise samples each with length {}".format(n,maxlen))
  X = get_random_one_hot(n, maxlen, len(char_indices))
  y = np.ones(n)

  history_a = model_a.fit(X, y,
    batch_size=128,
    epochs=epochs_a,
    callbacks=[print_callback])

  return { 
    'adversary': model_a,
    'discriminator': model_d,
    'history_d': history_d,
    'history_a': history_a
  }


def read_text():
  path = get_file('nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
  with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
  print('corpus length:', len(text))
  return text





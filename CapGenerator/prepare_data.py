from os import listdir
from pickle import dump
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Reshape, Concatenate
from tensorflow.keras.models import Model

from model import get_model

import argparse
import pandas as pd
import string

num_samples = 8091
batch_size = 128

# load an image from filepath
def image_gen(path, filenames):
  df = pd.DataFrame({'id': filenames})
  datagen = ImageDataGenerator(rescale=1. / 255)
  generator = datagen.flow_from_dataframe(
      dataframe=df,
      directory=path,
      x_col="id",
      target_size=(224, 224),
      batch_size=batch_size,
      class_mode=None,
      shuffle=False)
  return generator

# extract features from each photo in the directory
def extract_features(directory, model_type, is_attention, **kwargs):
  # load the model
  if is_attention:
    model = VGG16()
    model.layers.pop()
    # extract final 49x512 conv layer for context vectors
    final_conv = Reshape([49,512])(model.layers[-4].output)
    model = Model(inputs=model.inputs, outputs=final_conv)
  else:
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())

  # Load appropriate model
  model = get_model(model, model_type, **kwargs)
  # extract features from each photo
  feature_dict = dict()

  filenames = []
  for name in listdir(directory):
    # ignore README
    if name == 'README.md':
      continue
    filenames.append(name)

  # load image generator
  gen = image_gen(directory, filenames)
  print('Processing images in batches of size {}'.format(batch_size))
  # extract features
  features = model.predict(gen, num_samples, batch_size, verbose=1)

  for name, feature in zip(filenames, features):
    # get image id
    image_id = name.split('.')[0]
    # store feature
    feature_dict[image_id] = feature
  return feature_dict

# load doc into memory
def load_doc(filename):
  with open(filename, 'r') as f:
    # read all text
    text = f.read()
  return text

# extract descriptions for images
def load_descriptions(doc):
  mapping = dict()
  # process lines
  for line in doc.split('\n'):
    # split line by white space
    tokens = line.split()
    if len(line) < 2:
      continue
    # take the first token as the image id, the rest as the description
    image_id, image_desc = tokens[0], tokens[1:]
    # remove filename from image id
    image_id = image_id.split('.')[0]
    # convert description tokens back to string
    image_desc = ' '.join(image_desc)
    # create the list if needed
    if image_id not in mapping:
      mapping[image_id] = list()
    # store description
    mapping[image_id].append(image_desc)
  return mapping

def clean_descriptions(descriptions):
  # prepare translation table for removing punctuation
  table = str.maketrans('', '', string.punctuation)
  for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
      desc = desc_list[i]
      # tokenize
      desc = desc.split()
      # convert to lower case
      desc = [word.lower() for word in desc]
      # remove punctuation from each token
      desc = [w.translate(table) for w in desc]
      # remove hanging 's' and 'a'
      desc = [word for word in desc if len(word) > 1]
      # remove tokens with numbers in them
      desc = [word for word in desc if word.isalpha()]
      # store as string
      desc_list[i] = ' '.join(desc)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
  # build a list of all description strings
  all_desc = set()
  for key in descriptions.keys():
    [all_desc.update(d.split()) for d in descriptions[key]]
  return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
  lines = list()
  for key, desc_list in descriptions.items():
    for desc in desc_list:
      lines.append(key + ' ' + desc)
  data = '\n'.join(lines)
  with open(filename, 'w') as f:
    f.write(data)


def generate_features(model_type, **kwargs):
  # extract features from all images
  directory = 'Flickr8k_Dataset'
  features = extract_features(directory, model_type, is_attention=False, **kwargs)
  print('Extracted Features: %d' % len(features))
  # save to pickle file
  dump(features, open('models/features.pkl', 'wb'))

  # prepare descriptions
  filename = 'Flickr8k_text/Flickr8k.token.txt'
  # load descriptions
  doc = load_doc(filename)
  # parse descriptions
  descriptions = load_descriptions(doc)
  print('Loaded: %d ' % len(descriptions))
  # clean descriptions
  clean_descriptions(descriptions)
  # summarize vocabulary
  vocabulary = to_vocabulary(descriptions)
  print('Vocabulary Size: %d' % len(vocabulary))
  # save to file
  save_descriptions(descriptions, 'models/descriptions.txt')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate dataset image features')
  parser.add_argument("-t", "--type",
                      default='single',
                      help='Specify type of model.'
                           'Single GPU, Multi GPU or TPU')

  args = parser.parse_args()
  generate_features(args.type)

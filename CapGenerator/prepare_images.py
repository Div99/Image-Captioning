from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
import numpy as np
from progressbar import progressbar
from keras.models import Model

# load an image from filepath
def load_image(path):
    img = load_img(path, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return np.asarray(img)

# extract features from each photo in the directory
def extract_features(directory):
  # load the model
  model = VGG16()
  # re-structure the model
  model.layers.pop()
  model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
  print(model.summary())
  # extract features from each photo
  features = dict()

  for name in progressbar(listdir(directory)):
    # ignore README
    if name == 'README.md':
      continue
    filename = directory + '/' + name
    image = load_image(filename)
    # extract features
    feature = model.predict(image, verbose=0)
    # get image id
    image_id = name.split('.')[0]
    # store feature
    features[image_id] = feature
    print('>%s' % name)
  return features
# extract features from all images

directory = 'Flickr8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('models/features.pkl', 'wb'))

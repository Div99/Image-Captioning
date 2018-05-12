from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

import load_data as ld
import generate_model as gen
import argparse

# extract features from each photo in the directory
def extract_features(filename):
  # load the model
  model = VGG16()
  # re-structure the model
  model.layers.pop()
  model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
  # load the photo
  image = load_img(filename, target_size=(224, 224))
  # convert the image pixels to a numpy array
  image = img_to_array(image)
  # reshape data for the model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  # prepare the image for the VGG model
  image = preprocess_input(image)
  # get features
  feature = model.predict(image, verbose=0)
  return feature

# generate a description for an image
def generate_desc(model, tokenizer, photo, index_word, max_length):
  # seed the generation process
  in_text = 'startseq'
  # iterate over the whole length of the sequence
  for i in range(max_length):
    # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
    # predict next word
    yhat = model.predict([photo,sequence], verbose=0)
    # convert probability to integer
    yhat = argmax(yhat)
    # map integer to word
    word = index_word.get(yhat)
    # stop if we cannot map the word
    if word is None:
      break
    # append as input for generating the next word
    in_text += ' ' + word
    # stop if we predict the end of the sequence
    if word == 'endseq':
      break
  return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, index_word, max_length):
  actual, predicted = list(), list()
  # step over the whole set
  for key, desc_list in descriptions.items():
    # generate description
    yhat = generate_desc(model, tokenizer, photos[key], index_word, max_length)
    # store actual and predicted
    references = [d.split() for d in desc_list]
    actual.append(references)
    predicted.append(yhat.split())
  # calculate BLEU score
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))



if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Generate image captions')
  parser.add_argument("-i", "--image", help="Input image path")
  args = parser.parse_args()


  # load the tokenizer
  tokenizer = load(open('models/tokenizer.pkl', 'rb'))
  index_word = load(open('models/index_word.pkl', 'rb'))
  # pre-define the max sequence length (from training)
  max_length = 34
  # load the model
  filename = 'models/model-ep005-loss3.454-val_loss3.872.h5'
  model = load_model(filename)

  if args.image:
    # load and prepare the photograph
    photo = extract_features(args.image)
    # generate description
    description = generate_desc(model, tokenizer, photo, index_word, max_length)
    print(description)
  else:
    # load test set
    test_features, test_descriptions = ld.prepare_dataset('test')[1]

    # evaluate model
    evaluate_model(model, test_descriptions, test_features, tokenizer, index_word, max_length)

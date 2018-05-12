from pickle import load
import argparse

# load doc into memory
def load_doc(filename):
  # open the file as read only
  file = open(filename, 'r')
  # read all text
  text = file.read()
  # close the file
  file.close()
  return text

# load a pre-defined list of photo identifiers
def load_set(filename):
  doc = load_doc(filename)
  dataset = list()
  # process line by line
  for line in doc.split('\n'):
    # skip empty lines
    if len(line) < 1:
      continue
    # get the image identifier
    identifier = line.split('.')[0]
    dataset.append(identifier)
  return set(dataset)

# split a dataset into train/test elements
def train_test_split(dataset):
  # order keys so the split is consistent
  ordered = sorted(dataset)
  # return split dataset as two new sets
  return set(ordered[:100]), set(ordered[100:200])

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
  # load document
  doc = load_doc(filename)
  descriptions = dict()
  for line in doc.split('\n'):
    # split line by white space
    tokens = line.split()
    # split id from description
    image_id, image_desc = tokens[0], tokens[1:]
    # skip images not in the set
    if image_id in dataset:
      # create list
      if image_id not in descriptions:
        descriptions[image_id] = list()
      # wrap description in tokens
      desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
      # store
      descriptions[image_id].append(desc)
  return descriptions

# load photo features
def load_photo_features(filename, dataset):
  # load all features
  all_features = load(open(filename, 'rb'))
  # filter features
  features = {k: all_features[k] for k in dataset}
  return features

def prepare_dataset(data='dev'):

  assert data in ['dev', 'train', 'test']

  train_features = None
  train_descriptions = None

  if data == 'dev':
    # load dev set (1K)
    filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
    dataset = load_set(filename)
    print('Dataset: %d' % len(dataset))

    # train-test split
    train, test = train_test_split(dataset)
    #print('Train=%d, Test=%d' % (len(train), len(test)))

    # descriptions
    train_descriptions = load_clean_descriptions('models/descriptions.txt', train)
    test_descriptions = load_clean_descriptions('models/descriptions.txt', test)
    print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))

    # photo features
    train_features = load_photo_features('models/features.pkl', train)
    test_features = load_photo_features('models/features.pkl', test)
    print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))

  elif data == 'train':
    # load training dataset (6K)
    filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
    train = load_set(filename)

    filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
    test = load_set(filename)
    print('Dataset: %d' % len(train))

    # descriptions
    train_descriptions = load_clean_descriptions('models/descriptions.txt', train)
    test_descriptions = load_clean_descriptions('models/descriptions.txt', test)
    print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))

    # photo features
    train_features = load_photo_features('models/features.pkl', train)
    test_features = load_photo_features('models/features.pkl', test)
    print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))

  elif data == 'test':
    # load test set
    filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
    test = load_set(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    test_descriptions = load_clean_descriptions('models/descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    # photo features
    test_features = load_photo_features('models/features.pkl', test)
    print('Photos: test=%d' % len(test_features))

  return (train_features, train_descriptions), (test_features, test_descriptions)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate dataset features')
  parser.add_argument("-t", "--train", action='store_const', const='train',
    default = 'dev', help="Use large 6K training set")
  args = parser.parse_args()
  prepare_dataset(args.train)

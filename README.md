# Image-Captioning

Image Captioning System trained on Flick8k Dataset. 




Download the Flick8k Dataset here: [DataSet Request Form](https://illinois.edu/fb/sec/1713398)

----------------------------------

## Requirements 
- tensorflow
- keras
- numpy
- h5py
- progressbar2

These requirements can be easily installed by:
	`pip install -r requirements.txt`


## Scripts 

- __caption_generator.py__: The base script that contains functions for model creation, batch data generator etc.
- __prepare_descriptions.py__: Prepares the annotations for training. Changes have to be done to this script if new dataset is to be used. 
- __prepare_images.py__: Extracts features from images using VGG16 imagenet model. Changes have to be done to this script if new dataset is to be used. 
- __train_model.py__: Module for training the caption generator.
- __test_model.py__: Contains module for testing the performance of the caption generator, currently it contains the (BLEU)[https://en.wikipedia.org/wiki/BLEU] metric. New metrics can be added. 

## Usage

After the requirements have been installed, the process from training to testing is fairly easy. The commands to run:
1. `python prepare_images.py`
2. `python prepare_descriptions.py`
3. `python train_model.py`
4. `python test_model.py`

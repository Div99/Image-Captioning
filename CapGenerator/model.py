from tensorflow.keras.utils import multi_gpu_model
import numpy as np
import tensorflow as tf


class Multi_Model:

  def __init__(self, keras_model, type):
    self.type = type
    self.keras_model = keras_model

  def predict(self, generator, num_samples, batch_size, verbose):
    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on.
    if self.type == 'tpu':
      generator = tpu_gen(generator, num_samples % batch_size)
      features = self.keras_model.predict(generator, num_samples / batch_size, verbose=verbose)
      return features[:num_samples]

    else:
      return self.keras_model.predict(generator, num_samples / batch_size, verbose=verbose)


def tpu_gen(generator, dummy_indices):
  dummyImage = np.zeros((224, 224))
  for img in generator:
    yield img
  for k in dummy_indices:
    yield dummyImage


def get_model(model, type, **kwargs):
    if type == 'single':
        model = Multi_Model(model, 'single')

    elif type == 'multi':
        model = multi_gpu_model(model, gpus=2, cpu_relocation=True)
        model = Multi_Model(model, 'multi')

    elif type == 'tpu':
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(
                    kwargs['TPU_WORKER'])))
        model = Multi_Model(model, 'tpu')

    # tf.Keras requires models to be compiled even for pre-trained weights
    # (We just choose a random optimizer, doesn't affect prediction)
    model.keras_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

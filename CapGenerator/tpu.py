import tensorflow as tf


def tpu(model):
    tpu = tf.contrib.cluster_resolver.TPUClusterResolver()
    strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
    return model

from keras.utils import multi_gpu_model
import tensorflow as tf


def get_model(model, type, **args):
    if type == 'single':
        return model

    elif type == 'multi':
        return multi_gpu_model(model, gpus=2, cpu_relocation=True)

    elif type == 'tpu':
        tpu_model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(args['TPU_WORKER'])))
        return tpu_model

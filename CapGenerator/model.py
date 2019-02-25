from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf


def get_model(model, type, **kwargs):
    if type == 'single':
        pass

    elif type == 'multi':
        model = multi_gpu_model(model, gpus=2, cpu_relocation=True)

    elif type == 'tpu':
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(
                    kwargs['TPU_WORKER'])))

    # tf.Keras requires models to be compiled even for pre-trained weights
    # (We just choose a random optimizer, doesn't affect prediction)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

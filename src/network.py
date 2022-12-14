
import tensorflow as tf
from wavelet import Scattering

class Network(tf.keras.Model):
    def __init__(self,
                 layer_properties,
                 bins=128,
                 sampling_rate=1.0,
                 pool_type='avg',
                 data_format='channels_last',
                 combine=False,
                 name='scatnet'):
        super(Network, self).__init__(name=name)

        self.banks = [Scattering(bins=bins,
                                   sampling_rate=sampling_rate,
                                   **p) for p in layer_properties]
        self.sampling_rate = sampling_rate
        self.data_format = data_format
        self.combine = combine

        if pool_type == 'avg':
            self.pool = lambda x: tf.math.reduce_mean(x, axis=-1)
        elif pool_type == 'max':
            self.pool = lambda x: tf.math.reduce_max(x, axis=-1)
        else:
            self.pool = lambda x: x

    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs, _ = inputs
        x = inputs

        if self.data_format == 'channels_last':
            x = tf.transpose(x, [0, 2, 1])

        output = []

        for bank in self.banks:
            scalogram = bank(x)
            x = scalogram
            output.append(self.pool(scalogram))

        if self.combine:
            output = [tf.keras.layers.Reshape((-1, x.shape[-1]))(out) for out in output]
            output = tf.concat(output, axis=1)
            if self.data_format == 'channels_last':
                output = tf.transpose(output, [0, 2, 1])

        return output

    @property
    def depth(self):
        return len(self.banks)

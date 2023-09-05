
import tensorflow as tf
from tfscat.wavelet import Scattering

def median(x):
    mid = x.get_shape()[-1]//2 + 1
    return tf.math.top_k(x, mid).values[-1]

class Network(tf.keras.Model):
    def __init__(self,
                 layer_properties,
                 bins=128,
                 batch_size=32,
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
        self.pool_type = pool_type
        self.bins = bins
        self.batch_size = batch_size

        if pool_type == 'avg':
            self.pool = lambda x: tf.math.reduce_mean(x, axis=-1)
        elif pool_type == 'max':
            self.pool = lambda x: tf.math.reduce_max(x, axis=-1)
        elif pool_type == 'median':
            self.pool = lambda x: median(x)
        elif pool_type is None:
            self.pool = lambda x: x
        else:
            raise NotImplementedError(pool_type + ' no supported.')

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
            pooled = self.pool(scalogram)
            output.append(tf.squeeze(pooled))
            print(pooled.shape)

        if self.combine:
            output = [tf.reshape(out, (self.batch_size, -1, self.bins)) for out in output]
            output = tf.concat(output, axis=1)
            if self.data_format == 'channels_last' and self.pool_type is None:
                output = tf.transpose(output, [0, 2, 1])
                
        return output

    @property
    def depth(self):
        return len(self.banks)



import tensorflow as tf 
from scipy.signal.windows import tukey
import numpy as np
import math as m

class Scattering(tf.keras.layers.Layer):
    def __init__(self,
                 bins,
                 octaves=6,
                 resolution=1,
                 quality=4,
                 taper_alpha=1e-3,
                 sampling_rate=1.0,
                 trainable=False,
                 name='scattering'
                 ):

        super(Scattering, self).__init__(name=name)

        self.taper = tf.constant(tukey(bins, taper_alpha)[np.newaxis, np.newaxis, :], dtype=tf.float32)
        self.bins = bins
        self.octaves = octaves
        self.resolution = resolution
        self.quality = quality
        self.sampling_rate = sampling_rate
        self.trainable = trainable
        
        self.t = tf.Variable(self.times, trainable=self.trainable, name='times')
        self.w = tf.Variable(self.widths[:, None], trainable=self.trainable, name='widths')
        self.c = tf.Variable(self.centers[:, None], trainable=self.trainable, name='centers')
        self.filters = self.get_filters()

    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs, _ = inputs

        filters = self.get_filters() if self.trainable else self.filters
        sample = tf.signal.fft(tf.cast(inputs * self.taper, tf.complex64))
        convolved = tf.expand_dims(sample, axis=-2) * filters
        scalogram = tf.signal.fftshift(tf.signal.ifft(convolved), axes=-1)
        out = tf.math.abs(scalogram)

        return out

    def gaussian_window(self, x, width):
        return tf.math.exp(-((x / width) ** 2))

    def get_filters(self):

        if self.w.shape and self.c.shape:
            assert (
                    self.w.shape == self.c.shape
            ), f"Shape for widths {self.w.shape} and centers {self.c.shape} differ."

        s = tf.dtypes.complex(self.w, tf.zeros_like(self.w)) * tf.dtypes.complex(self.t, tf.zeros_like(self.t))
        c = tf.constant(2.0j, dtype=tf.complex64) * tf.constant(m.pi, dtype=tf.complex64)
        s = tf.math.exp(c * tf.cast(s, dtype=tf.complex64))
        w = self.gaussian_window(self.t, self.w)
        w = tf.cast(w, dtype=s.dtype)
        filters = w * s
        return tf.expand_dims(tf.signal.fft(filters), axis=0)

    @property
    def times(self):
        """Wavelet bank symmetric time vector in seconds."""
        duration = self.bins / self.sampling_rate
        return np.linspace(-0.5, 0.5, num=self.bins) * duration

    @property
    def frequencies(self):
        """Wavelet bank frequency vector in hertz."""
        return np.linspace(0, self.sampling_rate, self.bins)

    @property
    def nyquist(self):
        """Wavelet bank frequency vector in hertz."""
        return self.sampling_rate / 2

    @property
    def shape(self):
        """Filter bank total number of filters."""
        return self.octaves * self.resolution, self.bins

    @property
    def ratios(self):
        """Wavelet bank ratios."""
        ratios = np.linspace(self.octaves, 0.0, self.shape[0], endpoint=False)
        return -ratios[::-1]

    @property
    def scales(self):
        """Wavelet bank scaling factors."""
        return 2 ** self.ratios

    @property
    def centers(self):
        """Wavelet bank center frequencies."""
        return self.scales * self.nyquist

    @property
    def widths(self):
        """Wavelet bank temporal widths."""
        return self.quality / self.centers

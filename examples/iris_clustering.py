from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from sklearn.cluster import AgglomerativeClustering

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tfscat.network import Network
import time

def load_data():
    """Users: manage.
    
    Give traces as a numpy array.
    times : (n_times, )
    data: (n_channels, n_times)
    """
    
    client = Client("IRIS")
    
    stream = client.get_waveforms(network="YH",
                                  station='DC06',
                                  location='*',
                                  channel='*',
                                  starttime=UTCDateTime("2012-07-23T00:00:00"),
                                  endtime=UTCDateTime("2012-07-30T00:00:00"))
    stream.detrend("linear")
    stream.merge(method=1)
    stream.detrend("linear")
    stream.filter(type="highpass", freq=1)
    
    # Numpyification
    times = stream[0].times("matplotlib")
    data = np.array([trace.data for trace in stream])
        
    return times, data

T, X = load_data()
print(X.shape)

SEGMENT = 1200
STEP = 600
SAMPLING_RATE = 25
SAMPLES_PER_SEGMENT = int(SEGMENT * SAMPLING_RATE)
SAMPLES_PER_STEP = int(STEP * SAMPLING_RATE)
BANKS = (
    {"octaves": 4, "resolution": 4, "quality": 1},
    {"octaves": 9, "resolution": 1, "quality": 3}
)
feature_extractor = Sequential([Network(BANKS,
                                        bins=SAMPLES_PER_SEGMENT,
                                        sampling_rate=SAMPLING_RATE,
                                        pool_type='avg',
                                        data_format='channels_last',
                                        combine=True),
                                tf.keras.layers.Lambda(lambda x: tf.math.log(x + tf.keras.backend.epsilon()))])

dataset = tf.keras.utils.timeseries_dataset_from_array(X.T,
                                                       None, 
                                                       sequence_length=SAMPLES_PER_SEGMENT,
                                                       sequence_stride=SAMPLES_PER_STEP,
                                                       batch_size=32)

start = time.time()
p = feature_extractor.predict(dataset, verbose=1)
end = time.time()
print('Elapsed time:', end - start)

clusters = AgglomerativeClustering(5).fit_predict(p)

np.savez('tf/output/output.npz', cluster=clusters, times=T)


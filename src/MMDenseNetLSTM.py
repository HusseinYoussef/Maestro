
import tensorflow as tf
import feature_extraction as FS
import soundfile as sf
import numpy as np

class MMDenseNetLSTM:

    def batch_norm(self, x, train):
        return tf.keras.layers.BatchNormalization(epsilon=1e-5)(x,training=train)
        #return tf.keras.batch_normalization(x, momentum=0.99, epsilon=1e-5, training=train)

    def composite_layer(self, x, depth, train, bottle_neck=False):
        y = self.batch_norm(x, train)
        y = tf.nn.relu(y)
        siz = (3,3)
        factor = 1
        if bottle_neck:
            siz = (1, 1)
            factor = 4
        y = tf.keras.layers.Conv2D(filters= factor*depth, kernel_size= siz, padding= 'same', data_format='channels_first')(y)
        #y = tf.layers.conv2d(y, filters=factor*depth, kernel_size=siz, padding='same')
        return y

    def down_sample(self,x, depth):
        y = tf.keras.layers.Conv2D(filters= depth, kernel_size= (1,1), padding= 'same')(x)
        y = tf.keras.layers.AveragePooling2D(pool_size= (1,1), strides= (2,2))(y)
        return y

    def up_sample(self, x, depth):
        y = tf.keras.layers.Conv2DTranspose(filters=depth, kernel_size=(2,2), strides=(2,2), padding='same')(x)
        return y

    def dense_block(self, x, k, L, train):
        y_composite = x
        for _ in range(L):
            y_bottle = self.composite_layer(y_composite, k, train, bottle_neck=True)
            y_composite_new = self.composite_layer(y_bottle, k, train)
            y_composite = tf.concat([y_composite, y_composite_new], axis=-1)
        return y_composite_new

    def LSTM_layer(self, x, hidden_units):
        # x : [Batch, FrameAxis, FreqAxis, Channels]
        # y : [Batch, FrameAxis, FreqAxis]
        y = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), padding='same')(x)
        y = tf.squeeze(y, axis=3)
        lstm = tf.keras.layers.LSTM(units=hidden_units, return_sequences=True)
        #lstm = tf.compat.v1.keras.layers.CuDNNLSTM(units=hidden_units, return_sequence=True)
        y = lstm(y)
        y = tf.keras.layers.Dense(units= x.shape[-2])(y)
        return tf.expand_dims(y, axis=-1)

    def densenet_band1(self, x, name='densenet_band1', training=True, reuse=False):
        
        print(x.shape)
        # default params
        growth_rate = 14
        layers = 5

        with tf.compat.v1.variable_scope(name, reuse=reuse):
            # d1
            d1 = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=(3,3), padding='same',data_format='channels_first')(x)
            d1 = tf.cast(d1, np.float32)
            d1 = self.dense_block(d1, growth_rate, layers, training)
            print(d1.shape)
            # d2
            d2 = self.down_sample(d1, growth_rate)
            d2 = self.dense_block(d2, growth_rate, layers, training)
            print(d2.shape)
            # d3
            d3 = self.down_sample(d2, growth_rate)
            d3 = self.dense_block(d3, growth_rate, layers, training)
            print(d3.shape)
            # d4
            d4 = self.down_sample(d3, growth_rate)
            d4 = self.dense_block(d4, growth_rate, layers, training)
            d4_lstm = self.LSTM_layer(d4, 128)
            d4 = tf.concat([d4, d4_lstm], axis=-1)
            print(d4.shape)
            # u3
            u3 = self.up_sample(d4, growth_rate)
            u3 = self.dense_block(u3, growth_rate, layers, training)
            print(u3.shape)
            breakpoint()
            u3 = tf.concat([u3, d3], axis=-1)

            # u2
            u2 = self.up_sample(u3, growth_rate)
            u2 = self.dense_block(u2, growth_rate, layers, training)
            u2 = tf.concat([u2, d2], axis=-1)
            u2_lstm = self.LSTM_layer(u2, 128)
            u2 = tf.concat([u2, u2_lstm], axis=-1)

            # u1
            u1 = self.up_sample(u2, growth_rate)
            u1 = self.dense_block(u1, growth_rate, layers, training)
            u1 = tf.concat([u1, d1], axis=-1)

            return self.dense_block(u1, 12, 3, training)


    def densenet_band2(self, x, name='densenet_band2', training=True, reuse=False):

        # default params
        growth_rate = 4
        layers = 4

        with tf.compat.v1.variable_scope(name, reuse=reuse):
            # d1
            d1 = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=(3,3), padding='same')(x)
            d1 = self.dense_block(d1, growth_rate, layers, training)

            # d2
            d2 = self.down_sample(d1, growth_rate)
            d2 = self.dense_block(d2, growth_rate, layers, training)

            # d3
            d3 = self.down_sample(d2, growth_rate)
            d3 = self.dense_block(d3, growth_rate, layers, training)

            # d4
            d4 = self.down_sample(d3, growth_rate)
            d4 = self.dense_block(d4, growth_rate, layers, training)
            d4_lstm = self.LSTM_layer(d4, 32)
            d4 = tf.concat([d4, d4_lstm], axis=-1)

            # u3
            u3 = self.up_sample(d4, growth_rate)
            u3 = self.dense_block(u3, growth_rate, layers, training)
            u3 = tf.concat([u3, d3], axis=-1)

            # u2
            u2 = self.up_sample(u3, growth_rate)
            u2 = self.dense_block(u2, growth_rate, layers, training)
            u2 = tf.concat([u2, d2], axis=-1)

            # u1
            u1 = self.up_sample(u2, growth_rate)
            u1 = self.dense_block(u1, growth_rate, layers, training)
            u1 = tf.concat([u1, d1], axis=-1)

            return self.dense_block(u1, 12, 3, training)


    def densenet_band3(self, x, name='densenet_band3', training=True, reuse=False):

        # default params
        growth_rate = 2
        layers = 1

        with tf.compat.v1.variable_scope(name, reuse=reuse):
            # d1
            d1 = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=(3,3), padding='same')(x)
            d1 = self.dense_block(d1, growth_rate, layers, training)

            # d2
            d2 = self.down_sample(d1, growth_rate)
            d2 = self.dense_block(d2, growth_rate, layers, training)

            # d3
            d3 = self.down_sample(d2, growth_rate)
            d3 = self.dense_block(d3, growth_rate, layers, training)
            d3_lstm = self.LSTM_layer(d3, 8)
            d3 = tf.concat([d3, d3_lstm], axis=-1)

            # u2
            u2 = self.up_sample(d3, growth_rate)
            u2 = self.dense_block(u2, growth_rate, layers, training)
            u2 = tf.concat([u2, d2], axis=-1)

            # u1
            u1 = self.up_sample(u2, growth_rate)
            u1 = self.dense_block(u1, growth_rate, layers, training)
            u1 = tf.concat([u1, d1], axis=-1)

            return self.dense_block(u1, 12, 3, training)

    def densenet_full(self, x, name='densenet_full', training=True, reuse=False):

        # default params
        growth_rate = 7

        with tf.compat.v1.variable_scope(name, reuse=reuse):
            # d1
            d1 = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=(3,3), padding='same')(x)
            d1 = self.dense_block(d1, growth_rate, 3, training)

            # d2
            d2 = self.down_sample(d1, growth_rate)
            d2 = self.dense_block(d2, growth_rate, 3, training)

            # d3
            d3 = self.down_sample(d2, growth_rate)
            d3 = self.dense_block(d3, growth_rate, 4, training)

            # d4
            d4 = self.down_sample(d3, growth_rate)
            d4 = self.dense_block(d4, growth_rate, 5, training)
            d4_lstm = self.LSTM_layer(d4, 128)
            d4 = tf.concat([d4, d4_lstm], axis=-1)

            # d5
            d5 = self.down_sample(d4, growth_rate)
            d5 = self.dense_block(d5, growth_rate, 5, training)

            # u4
            u4 = self.up_sample(d5, growth_rate)
            u4 = self.dense_block(u4, growth_rate, 5, training)
            u4 = tf.concat([u4, d4], axis=-1)

            # u3
            u3 = self.up_sample(d4, growth_rate)
            u3 = self.dense_block(u3, growth_rate, 4, training)
            u3 = tf.concat([u3, d3], axis=-1)

            # u2
            u2 = self.up_sample(u3, growth_rate)
            u2 = self.dense_block(u2, growth_rate, 3, training)
            u2 = tf.concat([u2, d2], axis=-1)
            u2_lstm = self.LSTM_layer(u2, 128)
            u2 = tf.concat([u2, u2_lstm], axis=-1)

            # u1
            u1 = self.up_sample(u2, growth_rate)
            u1 = self.dense_block(u1, growth_rate, 3, training)
            u1 = tf.concat([u1, d1], axis=-1)

            return self.dense_block(u1, 12, 3, training)


    def forward(self, audios_mag, bands, training, reuse=False): # Training is bloolean.
        #NOTE: bands will be given in frequencies.
        # convert it to bins:
        #n_bins = 4096/2 + 1        # number of bins.
        #sample_rate = 44100

        # divide bands
        audios_band1 = audios_mag[ ..., bands[0]:bands[1]]
        audios_band2 = audios_mag[ : , : , bands[1]:bands[2]]
        audios_band3 = audios_mag[ : , : , bands[2]:bands[3]]
        audios_full  = audios_mag[ : , : , bands[0]:bands[3]]


        # densenet outputs
        outputs_band1 = self.densenet_band1(audios_band1)
        return 
        outputs_band2 = self.densenet_band2(audios_band2)
        outputs_band3 = self.densenet_band3(audios_band3)
        outputs_full = self.densenet_full(audios_full)

        # concat outputs along frequency axis
        outputs = tf.concat([outputs_band1, outputs_band2, outputs_band3], axis=2)
        # concat outputs along channel axis
        outputs = tf.concat([outputs, outputs_full], axis=3)

        # last conv to adjust channels
        outputs = tf.layers.conv2d(outputs, filters=2, kernel_size=[1, 2], padding='same')
        outputs = tf.concat([outputs, audios_mag[:, :, -1:]], axis=2)

        return outputs


if __name__ == "__main__":
    
    #model = MMDenseNetLSTM()
    sample_rate = 44100
    bands = [0, 384, 1024, 2048]

    mag, rate = sf.read('D:/CMP/4th/GP/Datasets[OUT]/musdb18/train/A Classic Education - NightOwl/mixture.wav')
    mag = mag[:8000]
    obj = FS.STFT()
    stft = obj(np.array([mag.T]))[-1]
    obj2 = FS.Spectrogram()
    spec = obj2(stft)
    
    #spec = tf.squeeze(spec,axis=1)
    breakpoint()
    print(spec.shape)
    spec = np.transpose(spec,(0,1,3,2))
    #model.forward(spec, bands,training=True)
    
    print("Hi")
    
    


import tensorflow as tf
import featurizer as FS
import soundfile as sf
import numpy as np
import utils 
from tensorflow.keras.models import Model
from tensorflow.keras.backend import expand_dims, squeeze, concatenate
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import ReLU, AveragePooling2D, Conv2DTranspose



class MMDenseNetLSTM:

    def composite_layer(self, x, depth, bottle_neck=False):
        y = BatchNormalization(momentum=0.99, epsilon=1e-5)(x)
        y = ReLU()(y)
        siz = 3
        factor = 1
        if bottle_neck:
            siz = 1
            factor = 4
        
        y = Conv2D(filters= factor*depth, kernel_size= siz, padding= 'same')(y)
        return y

    def down_sample(self,x, depth):
        y = Conv2D(filters= depth, kernel_size= (1,1), padding= 'same')(x)
        y = AveragePooling2D(pool_size= 2, strides= 2)(y)
        return y

    def up_sample(self, x, depth, wanted_shape):
        
        stride_size = 2
        filter_size = 2
        padding_size = 0  # valid.
        x_w = x.shape[1]
        x_h = x.shape[2]
        out_dim = lambda inp_dim: (inp_dim-1)*stride_size+filter_size-2*padding_size
        
        out_shape = (out_dim(x_w),out_dim(x_h))
        out_padding = tuple(map(lambda i, j: abs(i - j), wanted_shape, out_shape))
        y = Conv2DTranspose(filters=depth, kernel_size= filter_size, strides= stride_size, output_padding= out_padding)(x)
        return y

    def dense_block(self, x, k, L):
        y_composite = x
        for _ in range(L):
            y_bottle = self.composite_layer(y_composite, k, bottle_neck=True)
            y_composite_new = self.composite_layer(y_bottle, k)
            y_composite = concatenate([y_composite, y_composite_new], axis=-1)
        return y_composite_new

    def LSTM_layer(self, x, hidden_units):
        # x : [Batch, FrameAxis, FreqAxis, Channels]
        # y : [Batch, FrameAxis, FreqAxis]
        y = Conv2D(filters=1, kernel_size=(1,1), padding='same')(x)
        y = squeeze(y, axis=3)
        
        lstm = tf.keras.layers.LSTM(units=hidden_units, return_sequences=True)
        #lstm = tf.compat.v1.keras.layers.CuDNNLSTM(units=hidden_units, return_sequence=True)
        y = lstm(y)
        # y : [Batch, FrameAxis, FreqAxis]
        y = Dense(units= x.shape[-2])(y)
        # y : [Batch, FrameAxis, FreqAxis, 1]
        y = expand_dims(y, axis=-1)
        #y = Dense(units= x.shape[-1])(y)
        return y

    def densenet_band1(self, x, name= 'densenet_band1', log= False):

        if log : print("Constructing the first band")
        if log : print(x)
        # default params
        growth_rate = 14
        layers = 5
        
        # d1
        d1 = Conv2D(filters=growth_rate, kernel_size=3, padding='same', name= name)(x)
        if log : print(d1)
        d1 = self.dense_block(d1, growth_rate, layers)
        # d2
        d2 = self.down_sample(d1, growth_rate)
        if log : print(d2)
        d2 = self.dense_block(d2, growth_rate, layers)
        # d3
        d3 = self.down_sample(d2, growth_rate)
        if log : print(d3)
        d3 = self.dense_block(d3, growth_rate, layers)
        # d4
        d4 = self.down_sample(d3, growth_rate)
        if log : print(d4)
        d4 = self.dense_block(d4, growth_rate, layers)
        #breakpoint()
        d4_lstm = self.LSTM_layer(d4, 128)
        d4 = concatenate([d4, d4_lstm], axis=-1)

        # u3
        u3 = self.up_sample(d4, growth_rate, (d3.shape[1],d3.shape[2]))
        if log : print(u3)
        u3 = self.dense_block(u3, growth_rate, layers)
        u3 = concatenate([u3, d3], axis=-1)
        if log : print(u3)

        # u2
        u2 = self.up_sample(u3, growth_rate, (d2.shape[1],d2.shape[2]))
        if log : print(u2)
        u2 = self.dense_block(u2, growth_rate, layers)
        u2 = concatenate([u2, d2], axis=-1)
        u2_lstm = self.LSTM_layer(u2, 128)
        u2 = concatenate([u2, u2_lstm], axis=-1)
        if log : print(u2)

        # u1
        u1 = self.up_sample(u2, growth_rate, (d1.shape[1],d1.shape[2]))
        if log : print(u1)
        u1 = self.dense_block(u1, growth_rate, layers)
        u1 = concatenate([u1, d1], axis=-1)
        if log : print(u1)
        return self.dense_block(u1, 12, 3)

    def densenet_band2(self, x, name= 'densenet_band2', log= False):

        if log : print("constructing the second band")
        if log : print(x)

        # default params
        growth_rate = 4
        layers = 4

        # d1
        d1 = Conv2D(filters=growth_rate, kernel_size=3, padding='same',name= name)(x)
        if log : print(d1)
        d1 = self.dense_block(d1, growth_rate, layers)

        # d2
        d2 = self.down_sample(d1, growth_rate)
        if log : print(d2)
        d2 = self.dense_block(d2, growth_rate, layers)

        # d3
        d3 = self.down_sample(d2, growth_rate)
        if log : print(d3)
        d3 = self.dense_block(d3, growth_rate, layers)

        # d4
        d4 = self.down_sample(d3, growth_rate)
        if log : print(d4)
        d4 = self.dense_block(d4, growth_rate, layers)
        d4_lstm = self.LSTM_layer(d4, 32)
        d4 = concatenate([d4, d4_lstm], axis=-1)
        if log : print(d4)
        # u3
        u3 = self.up_sample(d4, growth_rate,(d3.shape[1],d3.shape[2]) )
        if log : print(u3)
        u3 = self.dense_block(u3, growth_rate, layers)
        u3 = concatenate([u3, d3], axis=-1)
        if log : print(u3)
        # u2
        u2 = self.up_sample(u3, growth_rate, (d2.shape[1],d2.shape[2]))
        if log : print(u2)
        u2 = self.dense_block(u2, growth_rate, layers)
        u2 = concatenate([u2, d2], axis=-1)
        if log : print(u2)

        # u1
        u1 = self.up_sample(u2, growth_rate, (d1.shape[1],d1.shape[2]))
        if log : print(u1)
        u1 = self.dense_block(u1, growth_rate, layers)
        u1 = concatenate([u1, d1], axis=-1)
        if log : print(u1)

        return self.dense_block(u1, 12, 3)


    def densenet_band3(self, x, name='densenet_band3', log= False):

        if log : print("constructing the third band")
        if log : print(x)

        # default params
        growth_rate = 2
        layers = 1
        # d1
        d1 = Conv2D(filters=growth_rate, kernel_size=3, padding='same')(x)
        if log : print(d1)
        d1 = self.dense_block(d1, growth_rate, layers)

        # d2
        d2 = self.down_sample(d1, growth_rate)
        if log : print(d2)
        d2 = self.dense_block(d2, growth_rate, layers)

        # d3
        d3 = self.down_sample(d2, growth_rate)
        if log : print(d3)
        d3 = self.dense_block(d3, growth_rate, layers)
        d3_lstm = self.LSTM_layer(d3, 8)
        d3 = concatenate([d3, d3_lstm], axis=-1)
        if log : print(d3)

        # u2
        u2 = self.up_sample(d3, growth_rate, (d2.shape[1],d2.shape[2]))
        if log : print(u2)
        u2 = self.dense_block(u2, growth_rate, layers)
        u2 = concatenate([u2, d2], axis=-1)
        if log : print(u2)

        # u1
        u1 = self.up_sample(u2, growth_rate,  (d1.shape[1],d1.shape[2]))
        if log : print(u1)
        u1 = self.dense_block(u1, growth_rate, layers)
        u1 = concatenate([u1, d1], axis=-1)
        if log : print(u1)

        return self.dense_block(u1, 12, 3)

    def densenet_full(self, x, name='densenet_full', log= False):

        if log : print("constructing the full band")
        if log : print(x)
        
        # default params
        growth_rate = 7

        
        # d1
        d1 = Conv2D(filters=growth_rate, kernel_size=3, padding='same')(x)
        if log : print(d1)
        d1 = self.dense_block(d1, growth_rate, 3)

        # d2
        d2 = self.down_sample(d1, growth_rate)
        if log : print(d2)
        d2 = self.dense_block(d2, growth_rate, 3)

        # d3
        d3 = self.down_sample(d2, growth_rate)
        if log : print(d3)
        d3 = self.dense_block(d3, growth_rate, 4)

        # d4
        d4 = self.down_sample(d3, growth_rate)
        if log : print(d4)
        d4 = self.dense_block(d4, growth_rate, 5)
        d4_lstm = self.LSTM_layer(d4, 128)
        d4 = tf.concat([d4, d4_lstm], axis=-1)
        if log : print(d4)

        # d5
        d5 = self.down_sample(d4, growth_rate)
        if log : print(d5)
        d5 = self.dense_block(d5, growth_rate, 5)

        # u4
        u4 = self.up_sample(d5, growth_rate,(d4.shape[1],d4.shape[2]))
        if log : print(u4)
        u4 = self.dense_block(u4, growth_rate, 5)
        u4 = tf.concat([u4, d4], axis=-1)
        if log : print(u4)

        # u3
        u3 = self.up_sample(d4, growth_rate,(d3.shape[1],d3.shape[2]))
        if log : print(u3)
        u3 = self.dense_block(u3, growth_rate, 4)
        u3 = tf.concat([u3, d3], axis=-1)
        if log : print(u3)

        # u2
        u2 = self.up_sample(u3, growth_rate, (d2.shape[1],d2.shape[2]))
        if log : print(u2)
        u2 = self.dense_block(u2, growth_rate, 3)
        u2 = tf.concat([u2, d2], axis=-1)
        u2_lstm = self.LSTM_layer(u2, 128)
        u2 = tf.concat([u2, u2_lstm], axis=-1)
        if log : print(u2)

        # u1
        u1 = self.up_sample(u2, growth_rate, (d1.shape[1],d1.shape[2]))
        if log : print(u1)
        u1 = self.dense_block(u1, growth_rate, 3)
        u1 = tf.concat([u1, d1], axis=-1)
        if log : print(u1)

        return self.dense_block(u1, 12, 3)


    def build(self, audios_mag, bands, log= False, summary= False):
        # audios_mag: the spectugram of the signal (frames, freq BINS, channels)
        # bands: array of size 4 contains the freqency bins limits for each band.

        assert(len(bands)==4)

        flatten_shape = 1
        for i in range(0,3):
            flatten_shape *= audios_mag.shape[i]
        
        inputs = Input(shape=(flatten_shape,))

        net = inputs
        net = Reshape(audios_mag.shape)(net)
        
        # divide bands
        band1_inp = net[ : , : ,bands[0]:bands[1], :]
        band2_inp = net[ : , : ,bands[1]:bands[2], :]
        band3_inp = net[ : , : ,bands[2]:bands[3], :]
        full_band_inp  = net[ : , : ,bands[0]:bands[3], :]
        
        # densenet outputs
        outputs_band1 = self.densenet_band1(band1_inp,log= log)
        print("Band1 Done.")
        outputs_band2 = self.densenet_band2(band2_inp,log = log)
        print("Band2 Done.")
        outputs_band3 = self.densenet_band3(band3_inp,log = log)
        print("Band3 Done.")
        outputs_full = self.densenet_full(full_band_inp,log = log)
        print("Full Band Done.")
        # concat outputs along frequency axis
        outputs = concatenate([outputs_band1, outputs_band2, outputs_band3], axis=2)
        if log: print(outputs)
        # concat outputs along channel axis
        outputs = concatenate([outputs, outputs_full], axis=3)
        if log: print(outputs)
        # last conv to adjust channels
        outputs = Conv2D(filters=2, kernel_size=[1, 2], padding='same')(outputs)
        if log: print(outputs)
        
        #outputs = concatenate([outputs, net[:, :, -1: ,:]], axis=2)  I think this is useless line.

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
        print('Model Compilation Done.')
        if summary : model.summary()

        return model


def bring_sample():
    mag, rate = utils.audio_loader('D:/CMP/4th/GP/Datasets[OUT]/musdb18/train/A Classic Education - NightOwl/mixture.wav')
    mag = mag[:8000]
    data = mag[None,...]
    #breakpoint()
    obj = FS.STFT()
    obj2 = obj(data)
    spec = FS.Spectrogram()
    spectugram = spec(obj2)
    sample = spectugram[0] #(n_channels, freq_bins, n_frames)
    sample = np.transpose(sample,(2,1,0))
    return sample


if __name__ == "__main__":
    
    
    sample_rate = 44100
    bands = [0, 385, 1025, 2049]
    sample = bring_sample() # (n_frames, freq_bins, n_channels)
    print(sample.shape)
    model = MMDenseNetLSTM()
    model.build(sample, bands,log= True)

    print("Hi")
    
    

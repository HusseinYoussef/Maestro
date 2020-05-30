
import tensorflow as tf
import numpy as np 
import math
import utils
import soundfile as sf
from featurizer import STFT, Spectrogram
from keras.backend import flatten, expand_dims
from glob import glob
import os
from tqdm import tqdm
import reconstruction
from tensorflow.keras.models import Model
from tensorflow.keras.backend import expand_dims, squeeze, concatenate
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import ReLU, AveragePooling2D, Conv2DTranspose



class MMDenseNetLSTM:

    def __calc_frames(self, samples):
        ''' Given number of samples, this function calculate the number of frames''' 
        if samples < self.frame_length:
            raise NameError(f'givrn number of samples =>{samples} is less than frame length')
        return (samples - self.frame_length) // self.frame_step + 1
        
    def __init__(self, seconds, frame_length= 4096, frame_step= 1024, sample_rate= 44100,freq_bands= 2049, channels= 2):
        ''' seconds argument is the time in seconds which the model will train and predict. '''
        self.seconds= seconds
        self.frame_length= frame_length
        self.frame_step= frame_step
        self.sample_rate= sample_rate
        self.freq_bands= freq_bands
        self.channels= channels
        self.frames = self.__calc_frames(self.seconds * self.sample_rate)


    def __composite_layer(self, x, depth, bottle_neck=False):
        y = BatchNormalization(momentum=0.99, epsilon=1e-5)(x)
        y = ReLU()(y)
        siz = 3
        factor = 1
        if bottle_neck:
            siz = 1
            factor = 4
        
        y = Conv2D(filters= factor*depth, kernel_size= siz, padding= 'same')(y)
        return y

    def __down_sample(self,x, depth):
        y = Conv2D(filters= depth, kernel_size= (1,1), padding= 'same')(x)
        y = AveragePooling2D(pool_size= 2, strides= 2)(y)
        return y

    def __up_sample(self, x, depth, wanted_shape):
        
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

    def __dense_block(self, x, k, L):
        y_composite = x
        for _ in range(L):
            y_bottle = self.__composite_layer(y_composite, k, bottle_neck=True)
            y_composite_new = self.__composite_layer(y_bottle, k)
            y_composite = concatenate([y_composite, y_composite_new], axis=-1)
        return y_composite_new

    def __LSTM_layer(self, x, hidden_units):
        # x : [Batch, FrameAxis, FreqAxis, Channels]
        # y : [Batch, FrameAxis, FreqAxis]
        y = Conv2D(filters=1, kernel_size=(1,1), padding='same')(x)
        y = squeeze(y, axis=3)
        
        lstm = tf.keras.layers.LSTM(units=hidden_units, return_sequences=True)
        y = lstm(y)
        # y : [Batch, FrameAxis, FreqAxis]
        y = Dense(units= x.shape[-2])(y)
        # y : [Batch, FrameAxis, FreqAxis, 1]
        y = expand_dims(y, axis=-1)
        #y = Dense(units= x.shape[-1])(y)
        return y

    def __densenet_band1(self, x, name= 'densenet_band1', log= False):

        if log : print("Constructing the first band")
        if log : print(x)
        # default params
        growth_rate = 14
        layers = 5
        
        # d1
        d1 = Conv2D(filters=growth_rate, kernel_size=3, padding='same', name= name)(x)
        if log : print(d1)
        d1 = self.__dense_block(d1, growth_rate, layers)
        # d2
        d2 = self.__down_sample(d1, growth_rate)
        if log : print(d2)
        d2 = self.__dense_block(d2, growth_rate, layers)
        # d3
        d3 = self.__down_sample(d2, growth_rate)
        if log : print(d3)
        d3 = self.__dense_block(d3, growth_rate, layers)
        # d4
        d4 = self.__down_sample(d3, growth_rate)
        if log : print(d4)
        d4 = self.__dense_block(d4, growth_rate, layers)
        #breakpoint()
        d4_lstm = self.__LSTM_layer(d4, 128)
        d4 = concatenate([d4, d4_lstm], axis=-1)

        # u3
        u3 = self.__up_sample(d4, growth_rate, (d3.shape[1],d3.shape[2]))
        if log : print(u3)
        u3 = self.__dense_block(u3, growth_rate, layers)
        u3 = concatenate([u3, d3], axis=-1)
        if log : print(u3)

        # u2
        u2 = self.__up_sample(u3, growth_rate, (d2.shape[1],d2.shape[2]))
        if log : print(u2)
        u2 = self.__dense_block(u2, growth_rate, layers)
        u2 = concatenate([u2, d2], axis=-1)
        u2_lstm = self.__LSTM_layer(u2, 128)
        u2 = concatenate([u2, u2_lstm], axis=-1)
        if log : print(u2)

        # u1
        u1 = self.__up_sample(u2, growth_rate, (d1.shape[1],d1.shape[2]))
        if log : print(u1)
        u1 = self.__dense_block(u1, growth_rate, layers)
        u1 = concatenate([u1, d1], axis=-1)
        if log : print(u1)
        return self.__dense_block(u1, 12, 3)

    def __densenet_band2(self, x, name= 'densenet_band2', log= False):

        if log : print("constructing the second band")
        if log : print(x)

        # default params
        growth_rate = 4
        layers = 4

        # d1
        d1 = Conv2D(filters=growth_rate, kernel_size=3, padding='same',name= name)(x)
        if log : print(d1)
        d1 = self.__dense_block(d1, growth_rate, layers)

        # d2
        d2 = self.__down_sample(d1, growth_rate)
        if log : print(d2)
        d2 = self.__dense_block(d2, growth_rate, layers)

        # d3
        d3 = self.__down_sample(d2, growth_rate)
        if log : print(d3)
        d3 = self.__dense_block(d3, growth_rate, layers)

        # d4
        d4 = self.__down_sample(d3, growth_rate)
        if log : print(d4)
        d4 = self.__dense_block(d4, growth_rate, layers)
        d4_lstm = self.__LSTM_layer(d4, 32)
        d4 = concatenate([d4, d4_lstm], axis=-1)
        if log : print(d4)
        # u3
        u3 = self.__up_sample(d4, growth_rate,(d3.shape[1],d3.shape[2]) )
        if log : print(u3)
        u3 = self.__dense_block(u3, growth_rate, layers)
        u3 = concatenate([u3, d3], axis=-1)
        if log : print(u3)
        # u2
        u2 = self.__up_sample(u3, growth_rate, (d2.shape[1],d2.shape[2]))
        if log : print(u2)
        u2 = self.__dense_block(u2, growth_rate, layers)
        u2 = concatenate([u2, d2], axis=-1)
        if log : print(u2)

        # u1
        u1 = self.__up_sample(u2, growth_rate, (d1.shape[1],d1.shape[2]))
        if log : print(u1)
        u1 = self.__dense_block(u1, growth_rate, layers)
        u1 = concatenate([u1, d1], axis=-1)
        if log : print(u1)

        return self.__dense_block(u1, 12, 3)


    def __densenet_band3(self, x, name='densenet_band3', log= False):

        if log : print("constructing the third band")
        if log : print(x)

        # default params
        growth_rate = 2
        layers = 1
        # d1
        d1 = Conv2D(filters=growth_rate, kernel_size=3, padding='same')(x)
        if log : print(d1)
        d1 = self.__dense_block(d1, growth_rate, layers)

        # d2
        d2 = self.__down_sample(d1, growth_rate)
        if log : print(d2)
        d2 = self.__dense_block(d2, growth_rate, layers)

        # d3
        d3 = self.__down_sample(d2, growth_rate)
        if log : print(d3)
        d3 = self.__dense_block(d3, growth_rate, layers)
        d3_lstm = self.__LSTM_layer(d3, 8)
        d3 = concatenate([d3, d3_lstm], axis=-1)
        if log : print(d3)

        # u2
        u2 = self.__up_sample(d3, growth_rate, (d2.shape[1],d2.shape[2]))
        if log : print(u2)
        u2 = self.__dense_block(u2, growth_rate, layers)
        u2 = concatenate([u2, d2], axis=-1)
        if log : print(u2)

        # u1
        u1 = self.__up_sample(u2, growth_rate,  (d1.shape[1],d1.shape[2]))
        if log : print(u1)
        u1 = self.__dense_block(u1, growth_rate, layers)
        u1 = concatenate([u1, d1], axis=-1)
        if log : print(u1)

        return self.__dense_block(u1, 12, 3)

    def __densenet_full(self, x, name='densenet_full', log= False):

        if log : print("constructing the full band")
        if log : print(x)
        
        # default params
        growth_rate = 7

        
        # d1
        d1 = Conv2D(filters=growth_rate, kernel_size=3, padding='same')(x)
        if log : print(d1)
        d1 = self.__dense_block(d1, growth_rate, 3)

        # d2
        d2 = self.__down_sample(d1, growth_rate)
        if log : print(d2)
        d2 = self.__dense_block(d2, growth_rate, 3)

        # d3
        d3 = self.__down_sample(d2, growth_rate)
        if log : print(d3)
        d3 = self.__dense_block(d3, growth_rate, 4)

        # d4
        d4 = self.__down_sample(d3, growth_rate)
        if log : print(d4)
        d4 = self.__dense_block(d4, growth_rate, 5)
        d4_lstm = self.__LSTM_layer(d4, 128)
        d4 = tf.concat([d4, d4_lstm], axis=-1)
        if log : print(d4)

        # d5
        d5 = self.__down_sample(d4, growth_rate)
        if log : print(d5)
        d5 = self.__dense_block(d5, growth_rate, 5)

        # u4
        u4 = self.__up_sample(d5, growth_rate,(d4.shape[1],d4.shape[2]))
        if log : print(u4)
        u4 = self.__dense_block(u4, growth_rate, 5)
        u4 = tf.concat([u4, d4], axis=-1)
        if log : print(u4)

        # u3
        u3 = self.__up_sample(d4, growth_rate,(d3.shape[1],d3.shape[2]))
        if log : print(u3)
        u3 = self.__dense_block(u3, growth_rate, 4)
        u3 = tf.concat([u3, d3], axis=-1)
        if log : print(u3)

        # u2
        u2 = self.__up_sample(u3, growth_rate, (d2.shape[1],d2.shape[2]))
        if log : print(u2)
        u2 = self.__dense_block(u2, growth_rate, 3)
        u2 = tf.concat([u2, d2], axis=-1)
        u2_lstm = self.__LSTM_layer(u2, 128)
        u2 = tf.concat([u2, u2_lstm], axis=-1)
        if log : print(u2)

        # u1
        u1 = self.__up_sample(u2, growth_rate, (d1.shape[1],d1.shape[2]))
        if log : print(u1)
        u1 = self.__dense_block(u1, growth_rate, 3)
        u1 = tf.concat([u1, d1], axis=-1)
        if log : print(u1)

        return self.__dense_block(u1, 12, 3)


    def build(self, bands= [0, 385, 1025, 2049], log= False, summary= False):
        ''' bands: array of size 4 contains the freqency bins limits for each band. '''

        assert(len(bands)==4)

        flatten_shape = self.freq_bands * self.frames * self.channels
        #sample = np.random.rand(frames, freq_bands, channels)
        
        inputs = Input(shape=(flatten_shape,))

        net = inputs
        net = Reshape( (self.frames, self.freq_bands, self.channels) )(net)
        
        # divide bands
        band1_inp = net[ : , : ,bands[0]:bands[1], :]
        band2_inp = net[ : , : ,bands[1]:bands[2], :]
        band3_inp = net[ : , : ,bands[2]:bands[3], :]
        full_band_inp  = net[ : , : ,bands[0]:bands[3], :]
        
        # densenet outputs
        outputs_band1 = self.__densenet_band1(band1_inp,log= log)
        print("Band1 Done.")
        outputs_band2 = self.__densenet_band2(band2_inp,log = log)
        print("Band2 Done.")
        outputs_band3 = self.__densenet_band3(band3_inp,log = log)
        print("Band3 Done.")
        outputs_full = self.__densenet_full(full_band_inp,log = log)
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
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        print('Model Compilation Done.')
        if summary : model.summary()

        return model
    
    
    def __prepare(self, track):
        ''' Tranform the track into spectrogram, reshape it and return it '''
        return np.transpose((Spectrogram())((STFT())(track[None,...]))[0],(2,1,0))


    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, labels, x_path, y_path, dim, shuffle= True, batch_size= 1):
            self.labels = labels # names of the tracks in the dataset. NOTE: names of X is same as Y.
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.x_path = x_path
            self.y_path = y_path
            self.dim = dim
            self.on_epoch_end()

        def __len__(self):
            ''' The number of batches per epoch '''
            return int(np.floor(len(self.labels) / self.batch_size))
        
        def __getitem__(self, index):
            ''' Generate one batch of data '''
            return self.__data_generation(self.labels[index * self.batch_size : (index+1) * self.batch_size])
        
        def on_epoch_end(self):
            if self.shuffle == True: np.random.shuffle(self.labels)
        
        def __data_generation(self, current_labels):
            
            _x = np.empty( (self.batch_size, self.dim[0]*self.dim[1]*self.dim[2]) )
            _y = np.empty( (self.batch_size, *self.dim) )

            for i, ID in enumerate(current_labels):
                _x[i,] = flatten(np.load(f'{self.x_path}/{ID}.npy'))
                _y[i,] = np.load(f'{self.y_path}/{ID}.npy')
            return _x, _y

    def __clean(self, labels):
        for i,ID in enumerate(labels):
            labels[i] = ''.join(filter(str.isdigit, ID))
        return labels

    def Train(self, model_directory, # the path to the model directory. (where the model will be saved/loaded).
            resume, # boolean to check whether the training will start from a checkpoint or from the begining.
            epochs, # number of epochs (must be > init_epochs)
            batch_size, # dimensions
            train_X_path, # directory.
            train_Y_path, # directory.
            valid_X_path, # directory.
            valid_Y_path, # directory.
            logs_path, # directory.
            init_epochs= 0, # number of initial epochs (epochs which is done.)
            evaluate= False, # boolean to evaluate the model after training. the evaluation is done using the test set.
            test_X_path= None,
            test_Y_path = None): 
        ''' Before using this function, there should be directories containing the splitted tracks spectrugrams. '''
        ''' This function returns 2 things: history , evalution '''
        
        
        # Initialize Callbacks:
        mc = tf.keras.callbacks.ModelCheckpoint(filepath=f'{model_directory}/model.keras', monitor='val_mean_squared_error', mode='min', verbose=1, save_best_only=True)
        mp = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, embeddings_freq=0, update_freq="epoch")
        sp = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=10)
        initial_epoch = init_epochs

        model = tf.keras.models.load_model(model_directory) if resume else self.build()

        # Start/resume training
        train_data = glob(train_X_path + '/*')
        valid_data = glob(valid_X_path + '/*')
        

        train_data = self.__clean(train_data) # pure IDs for training items.
        valid_data = self.__clean(valid_data) # pure IDs for validation items.

        training_generator = self.DataGenerator(train_data, dim= (self.frames, self.freq_bands, 2), x_path= train_X_path, y_path= train_Y_path, batch_size= batch_size)
        validation_generator = self.DataGenerator(valid_data, dim= (self.frames, self.freq_bands, 2), x_path= valid_X_path, y_path= valid_Y_path, batch_size= batch_size)

        history = model.fit(training_generator,
                            validation_data= validation_generator,
                            workers= 8, verbose= 1, epochs= epochs,
                            callbacks= [mc, mp, sp], initial_epoch= initial_epoch)
        evaluation = None
        if evaluate:
            print('\n\nevaluation\n\n')
            test_data = glob(test_X_path + '/*')
            test_data = self.clean(test_data)
            testing_generator = DataGenerator(test_data, dim= (self.frames, self.freq_bands, 2), x_path= test_X_path, y_path= test_Y_path, batch_size= batch_size)
            evaluation = model.evaluate(testing_generator,
                        batch_size=batch_size,
                        verbose=1, workers=8, return_dict=True)

        return history, evaluation
    
    def Predict(self, model, # can be the model itself or model path.
            track, # can be the track itself (samples, channels) or track path.
            output_directory, track_name):

        ''' this function takes a track whatever its length and then predict its stem in the output_directory '''
  
        if type(model) == str: # if model is a path not real object.
            if not os.path.exists(model):
                raise NameError('Model does not exist')
            model = tf.keras.models.load_model(model)

        if type(track) == str: # if track is a path not real object.
            if not os.path.exists(track):
                raise NameError('Track does not exist')
            track, sample_rate = sf.read(track, always_2d=True)
        
        # TODO check if the track is stereo or mono.
        wanted_frames = int(math.ceil(self.__calc_frames(len(track)) / self.frames )) * self.frames
        wanted_len = (wanted_frames - 1) * self.frame_step + self.frame_length

        assert(self.__calc_frames(wanted_len) == wanted_frames)

        padding = wanted_len - len(track)
        track = np.append(track,padding*[[0,0]], axis=0)

        assert(len(track) == wanted_len)
        
        mix_spec = self.__prepare(track.T)
        iterations = self.__calc_frames(len(track)) // self.frames
        mix_spec = np.reshape(mix_spec, ( mix_spec.shape[0] // self.frames, self.frames, mix_spec.shape[1], mix_spec.shape[2]) )

        assert(mix_spec.shape[1] == self.frames)
        assert(mix_spec.shape[2] == self.freq_bands)
        assert(mix_spec.shape[3] == self.channels)

        full_output_stem = None
        
        for i in tqdm(np.arange(iterations), total= iterations):
            print(mix_spec[i].shape)
            output = model.predict(expand_dims(flatten(mix_spec[i]),0))
            output = output[0]
            full_output_stem = np.concatenate([full_output_stem,output],axis=0) if i>0 else output
            
        #breakpoint()
        track = track.T # (channel, samples)
        stft_mix = np.transpose(((STFT())(track[None,...]))[0],(2,1,0)) # the stft which is needed for reconstruction.
        
        predicted_stem = reconstruction.reconstruct( np.expand_dims(full_output_stem, axis= -1), stft_mix, ['drums'], boundary= False)['drums']
        predicted_stem = predicted_stem[:-padding , :] # remove padding. (samples, channels)

        track = track.T
        track = track[:-padding, :] # remove padding. (samples, channels)

        print('prediction done.')
        sf.write(f'{output_directory}/{track_name}[stem].wav', predicted_stem, self.sample_rate)
        sf.write(f'{output_directory}/{track_name}[mix].wav', track, self.sample_rate)
        utils.convert_to_mp3(f'{output_directory}/{track_name}[stem].wav', f'{output_directory}/{track_name}[stem].mp3')
        utils.convert_to_mp3(f'{output_directory}/{track_name}[mix].wav', f'{output_directory}/{track_name}[mix].mp3')
        print('writing done.')
        

if __name__ == "__main__":
    
    mix_path = 'D:/CMP/4th/GP/Test/Buitraker - Revo X/mixture.wav'
    model_path = 'D:/CMP/4th/GP/Test/check_point'
    model = MMDenseNetLSTM(seconds= 3)
    model.Predict(model= model_path, track= mix_path, output_directory= 'D:/CMP/4th/GP/Test/', track_name= 'AM Contra - Heart Peripheral')
    '''
    sample_rate = 44100
    bands = [0, 385, 1025, 2049]
    #sample = bring_sample() # (n_frames, freq_bins, n_channels)
    #print(sample.shape)
    model = MMDenseNetLSTM()
    model.build(2049, 5000, 2, bands,log= True)
    '''
    
    
     


    print("Hi")
    
    

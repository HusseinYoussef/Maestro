
import tensorflow as tf
import numpy as np 
import utils
import soundfile as sf
from dataset import split_track
from featurizer import STFT, Spectrogram
from keras.backend import flatten, expand_dims
from glob import glob
from tensorflow.keras.models import Model
from tensorflow.keras.backend import expand_dims, squeeze, concatenate
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import ReLU, AveragePooling2D, Conv2DTranspose



class MMDenseNetLSTM:
    def MMDenseNetLSTM(self, seconds, frame_length= 4096, frame_step= 1025, freq_rate= 44100,freq_bands= 2049, channels= 2):
        ''' seconds argument is the time in seconds which the model will train and predict. '''
        self.seconds= seconds
        self.frame_length= frame_length
        self.frame_step= frame_step
        self.freq_rate= freq_rate
        self.freq_bands= freq_bands
        self.channels= channels
        samples = self.seconds * self.freq_rate
        self.frames = (samples - self.frame_length) // self.frame_step + 1


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

        flatten_shape = self.freq_bins * self.frames * self.channels
        #sample = np.random.rand(frames, freq_bins, channels)
        
        inputs = Input(shape=(flatten_shape,))

        net = inputs
        net = Reshape( (self.frames, self.freq_bins, self.channels) )(net)
        
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
            
                p1 = f'{self.x_path}/{ID}.wav'
                p2 = f'{self.y_path}/{ID}.wav'

                mix, rate = utils.audio_loader(p1)
                stem, rate = utils.audio_loader(p2)
                _x[i,] = flatten(self.prepare(mix))
                _y[i,] = self.prepare(stem)
            
            return _x, _y

    def __clean(self, labels):
        for i,ID in enumerate(labels):
            labels[i] = ''.join(filter(str.isdigit, ID))
        return labels

    def Train(self, model_directory, # the path to the model directory. (where the model will be saved/loaded).
            resume, # boolean to check whether the training will start from a checkpoint or from the begining.
            init_epochs, # number of initial epochs (epochs which is done.)
            epochs, # number of epochs (must be > init_epochs)
            batch_size, # dimensions
            freq_bands, # dimensions
            frames, # dimensions
            train_X_path, # directory.
            train_Y_path, # directory.
            valid_X_path, # directory.
            valid_Y_path, # directory.
            logs_path, # directory.
            evaluate= False, # boolean to evaluate the model after training. the evaluation is done using the test set.
            test_X_path= None,
            test_Y_path = None): 
        ''' Before using this function, there should be directories containing the splitted tracks. '''
        ''' This function returns 2 things: history , evalution '''
        
        
        # Initialize Callbacks:
        mc = tf.keras.callbacks.ModelCheckpoint(filepath=model_directory, monitor='val_mean_squared_error', mode='min', verbose=1, save_best_only=True)
        mp = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, embeddings_freq=0, update_freq="epoch")
        initial_epoch = 0
        # Load checkpoint:
        if resume:
            # Load model:
            model = tf.python.keras.models.load_model(model_directory)
            #initial_epoch = int (((model_path.split('/')[-1]).split('.')[1]).split('-')[0]) + 1
            initial_epoch = init_epochs
        else:
            model = MMDenseNetLSTM().build(freq_bins=freq_bands, frames= frames, channels= 2, bands= [0, 385, 1025, 2049])
        # Start/resume training
        train_data = glob(train_X_path + '/*')
        valid_data = glob(valid_X_path + '/*')
        

        train_data = self.clean(train_data)
        valid_data = self.clean(valid_data)

        training_generator = DataGenerator(train_data, dim= (frames, freq_bands, 2), x_path= train_X_path, y_path= train_Y_path, batch_size= batch_size)
        validation_generator = DataGenerator(valid_data, dim= (frames, freq_bands, 2), x_path= valid_X_path, y_path= valid_Y_path, batch_size= batch_size)

        history = model.fit(training_generator,
                            validation_data= validation_generator,
                            workers= 8, verbose= 1, epochs= epochs,
                            callbacks= [mc, mp], initial_epoch= initial_epoch)
        evaluation = None
        if evaluate:
            print('\n\nevaluation\n\n')
            test_data = glob(test_X_path + '/*')
            test_data = self.clean(test_data)
            testing_generator = DataGenerator(test_data, dim= (frames, freq_bands, 2), x_path= test_X_path, y_path= test_Y_path, batch_size= batch_size)
            evaluation = model.evaluate(testing_generator,
                        batch_size=batch_size,
                        verbose=1,
                        workers=8,
                        return_dict=True)

        return history, evaluation
    
    def Predict(self, model, # can be the model itself or model path.
            track, # can be the track itself (samples, channels) or track path.
            output_directory, track_name):

        ''' this function takes a track whatever its length and then predict its stem in the output_directory '''
  
        if type(model) == str: # if model is a path not real object.
            model = tf.python.keras.models.load_model(model)

        if type(track) == str: # if track is a path not real object.
            track, sample_rate = sf.read(track, always_2d=True)

        # expand the track then split it.

        siz = len(track) # number of samples.
        divisor = self.sample_rate * self.seconds # convert seconds to samples.
        new_siz = (siz//divisor + 1) * divisor
        reminder = int((new_siz-siz) % divisor) # reminder is used to remove the added part to the track.
        splitted = split_track(track, self.freq_rate, self.seconds)  # array of splitted tracks.
  
        full_output = None
        flag = False

        obj = STFT()
        for part in splitted:
            part = part.T
            output = model.predict(expand_dims(flatten(self.__prepare(part)),0))
            output = np.transpose(output[0], (0,2,1))
            if flag == False:
                full_output = output
                flag = True
            else:
                full_output = np.concatenate([full_output,output],axis=0) # concate on frames.
    
        ''' TODO: the remaining part is to reconstruct the track then save it in the given path. '''
        #predicted_stem = reconstruct()

if __name__ == "__main__":
    
    '''
    sample_rate = 44100
    bands = [0, 385, 1025, 2049]
    #sample = bring_sample() # (n_frames, freq_bins, n_channels)
    #print(sample.shape)
    model = MMDenseNetLSTM()
    model.build(2049, 5000, 2, bands,log= True)
    '''
    
    
     


    print("Hi")
    
    

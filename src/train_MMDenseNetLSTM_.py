import numpy as np
import utils
from featurizer import STFT, Spectrogram
from keras.backend import flatten, expand_dims
from MMDenseNetLSTM import MMDenseNetLSTM
import tensorflow as tf
from glob import glob
import os

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

  def prepare(self, track):
    ''' Tranform the track into spectrogram, reshape it and return it '''
    return np.transpose(Spectrogram(STFT(track[None,...]))[0],(2,1,0))
  
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

def clean(labels):
  for i,ID in enumerate(labels):
    IDs[i] = ''.join(filter(str.isdigit, ID))
  return labels

def Train(model_path, freq_bands, frames, train_X_path, train_Y_path, valid_X_path, valid_Y_path):
    '''
    model_path = model directory
    any path must be sent without the last slash "/"
    ''' 

    if not( os.path.exists(model_path) ):
        os.mkdir(model_path)
    
    model_path += 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    # Initialize Callbacks:
    mc = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_mse', mode='min', verbose=1, save_best_only=True)

    # Load checkpoint:
    if model_path is not None:
        # Load model:
        model = tf.python.keras.models.load_model(model_path)
    else:
        model = MMDenseNetLSTM().build(freq_bins=freq_bands, frames= frames, channels= 2, bands= [0, 385, 1025, 2049])
    # Start/resume training
    train_data = glob(train_X_path + '/*')
    valid_data = glob(valid_X_path + '/*')

    train_data = clean(train_data)
    valid_data = clean(valid_data)

    training_generator = DataGenerator(train_data, (frames, freq_bands, 2), x_path= train_X_path, y_path= train_Y_path, batch_size= 10)
    validation_generator = DataGenerator(valid_data, (frames, freq_bands, 2), x_path= valid_X_path, y_path= valid_Y_path, batch_size= 10)

    return model.fit(training_generator,validation_data= validation_generator, workers= 6, verbose= 1, epochs= 1, callbacks= [mc], initial_epoch= 0)

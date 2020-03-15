import tensorflow as tf

class MMDenseNetLSTM:

    def batch_norm(x, train):   
        return tf.layers.batch_normalization(x, momentum=0.99, epsilon=1e-5, training = train)

    def composite_layer(x, depth, train, bottle_neck=False):
        y = batch_norm(x, train)
        y = tf.nn.relu(y)
        siz = [3,3]
        factor = 1
        if bottle_neck:
            siz = [1,1]
            factor=4

        y = tf.layers.conv2d(y, filters= factor*depth, kernel_size=siz, padding='same')
        return y

    def down_sample(x, depth):
        y = tf.layers.conv2d(x, filters= depth, kernel_size=[1,1], padding='same')
        y = tf.layers.average_pooling2d(x, pool_size=[1,1], strides=[2,2])
        return y

    def up_sample(x, depth):
        y = tf.layers.conv2d_transpose(x, filters=depth, kernel_size=[2, 2], strides=[2, 2], padding='same')
        return y
    
    def dense_block(x, k, L, train):
        y_composite = x
        for _ in range(L):
            y_bottle = composite_layer(y_composite, k, train, True)
            y_composite_new = composite_layer(y_bottle, k, train) 
            y_composite = tf.concat([y_composite, y_composite_new], axis=-1)
        return y_composite_new

    def LSTM_layer(x, hidden_units):
        # x : [Batch, FrameAxis, FreqAxis, Channels]
        # y : [Batch, FrameAxis, FreqAxis]
        y = tf.layers.conv2d(x, filters=1, kernel_size=[1,1], padding='same')
        y = tf.squeeze(y, axis=3)
        lstm = tf.keras.layers.CuDNNLSTM(units=hidden_units, return_sequence=True)
        y = lstm(y)
        y = tf.layers.dense(y, x.shape[-2])
        return tf.expand_dims(y, axis=-1)

    


        


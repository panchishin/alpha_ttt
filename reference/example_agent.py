import tensorflow as tf
from tensorflow import keras

def agent(state_shape, action_shape, learning_rate=0.001):
    
    init = tf.keras.initializers.HeUniform()

    model = keras.Sequential((
    	keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init),
    	keras.layers.Dense(12, activation='relu', kernel_initializer=init),
    	keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init),
		))

    model.compile(
		loss=tf.keras.losses.Huber(), 
		optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
		metrics=['accuracy'])

    return model
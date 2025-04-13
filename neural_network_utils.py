#!/usr/bin/env python3
# Defines neural network model and helpers

import tensorflow as tf

class FeatureProviderLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_features(self, features):
        self.features = tf.constant(features)

    def call(self, index):
        return tf.gather(self.features, index)

    def get_config(self):
        return super().get_config()

class ComplexToRealImagLayer(tf.keras.layers.Layer):
    def __init__(self, dtype):
        super(ComplexToRealImagLayer, self).__init__(dtype = dtype)

    def call(self, features):
        return tf.stack([tf.math.real(features), tf.math.imag(features)], axis = -1)

def construct_model(input_shape, name, dtype = tf.complex64):
	return tf.keras.models.Sequential([
		tf.keras.Input(shape = input_shape, name = "input", dtype = dtype),
		ComplexToRealImagLayer(dtype = dtype),
		tf.keras.layers.Flatten(),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(1024, activation = "relu"),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(512, activation = "relu"),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(256, activation = "relu"),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(128, activation = "relu"),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(64, activation = "relu"),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(2, activation = "linear"),
	], name = name)

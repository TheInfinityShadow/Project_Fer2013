# src/cbam.py
from tensorflow.keras import layers
import tensorflow as tf

class CBAMLayer(layers.Layer):
    def __init__(self, ratio=8):
        super(CBAMLayer, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        self.channel = input_shape[-1]
        self.shared_dense_one = layers.Dense(self.channel // self.ratio, activation='relu')
        self.shared_dense_two = layers.Dense(self.channel)
        self.spatial_conv = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')

    def call(self, input_tensor):
        avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
        max_pool = layers.GlobalMaxPooling2D()(input_tensor)

        avg = self.shared_dense_one(avg_pool)
        avg = self.shared_dense_two(avg)

        max = self.shared_dense_one(max_pool)
        max = self.shared_dense_two(max)

        channel_attention = layers.Activation('sigmoid')(layers.Add()([avg, max]))
        channel_attention = layers.Reshape((1, 1, self.channel))(channel_attention)
        x = layers.Multiply()([input_tensor, channel_attention])

        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_attention = self.spatial_conv(layers.Concatenate()([avg_pool, max_pool]))

        return layers.Multiply()([x, spatial_attention])

    def get_config(self):
        config = super(CBAMLayer, self).get_config()
        config.update({"ratio": self.ratio})
        return config
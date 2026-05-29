import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="Attention")
class cSE(layers.Layer):
    def __init__(self, ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(1, int(channels * self.ratio))
        self.gap = layers.GlobalAveragePooling2D()
        self.reshape = layers.Reshape((1, 1, channels))
        self.fc1 = layers.Dense(hidden, activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")
        self.mul = layers.Multiply()
        super().build(input_shape)

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.reshape(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.mul([inputs, x])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio})
        return cfg

@tf.keras.utils.register_keras_serializable(package="Attention")
class sSE(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")
        self.mul = layers.Multiply()
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        return self.mul([inputs, x])

@tf.keras.utils.register_keras_serializable(package="Attention")
class scSE(layers.Layer):
    def __init__(self, ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.cse = cSE(ratio=self.ratio)
        self.sse = sSE()
        self.add = layers.Add()
        super().build(input_shape)

    def call(self, inputs):
        x1 = self.cse(inputs)
        x2 = self.sse(inputs)
        return self.add([x1, x2])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio})
        return cfg

@tf.keras.utils.register_keras_serializable(package="Attention")
class ChannelReduce(layers.Layer):
    def __init__(self, mode="max", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def call(self, inputs):
        if self.mode == "max":
            return tf.reduce_max(inputs, axis=-1, keepdims=True)
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"mode": self.mode})
        return cfg

@tf.keras.utils.register_keras_serializable(package="Attention")
class CBAM(layers.Layer):
    def __init__(self, ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = int(input_shape[-1])
        hidden = max(1, int(channel * self.ratio))

        # Channel Attention Parts
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        self.reshape = layers.Reshape((1, 1, channel))
        self.shared_dense1 = layers.Dense(hidden, activation="relu")
        self.shared_dense2 = layers.Dense(channel)

        # Spatial Attention Parts
        self.max_reduce = ChannelReduce("max")
        self.mean_reduce = ChannelReduce("mean")
        self.spatial_conv = layers.Conv2D(1, kernel_size=7, padding="same", use_bias=False)
        
        self.sigmoid = layers.Activation("sigmoid")
        self.add = layers.Add()
        self.mul = layers.Multiply()
        self.concat = layers.Concatenate(axis=-1)

        super().build(input_shape)

    def call(self, inputs):
        # Channel Attention
        avg_pool = self.reshape(self.gap(inputs))
        max_pool = self.reshape(self.gmp(inputs))
        
        avg_out = self.shared_dense2(self.shared_dense1(avg_pool))
        max_out = self.shared_dense2(self.shared_dense1(max_pool))
        
        channel_att = self.sigmoid(self.add([avg_out, max_out]))
        x = self.mul([inputs, channel_att])

        # Spatial Attention
        s_max = self.max_reduce(x)
        s_avg = self.mean_reduce(x)
        spatial_att = self.concat([s_max, s_avg])
        spatial_att = self.spatial_conv(spatial_att)
        spatial_att = self.sigmoid(spatial_att)

        return self.mul([x, spatial_att])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio})
        return cfg
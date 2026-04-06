import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Create trainable weights."""
        # W: Weight matrix
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )

        # b: Bias vector
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )

        # u: Context vector
        self.u = self.add_weight(
            name='attention_context',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        """Forward pass."""
        # Compute attention scores
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)

        # Convert to weights
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, -1)

        # Weighted sum
        weighted_input = x * ait
        output = tf.reduce_sum(weighted_input, axis=1)

        return output

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


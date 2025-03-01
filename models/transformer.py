# transformer.py
import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles
        
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices
        sines = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices
        cosines = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = np.zeros(angle_rads.shape)
        pos_encoding[:, 0::2] = sines
        pos_encoding[:, 1::2] = cosines
        
        pos_encoding = pos_encoding[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def build_transformer_model(input_shape, output_dim=1):
    """
    Build Transformer model for time series prediction
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Positional encoding
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
    
    # Transformer layers
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=8, key_dim=input_shape[1]//8
    )(x, x)
    
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # Feed-forward network
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(input_shape[1])
    ])
    
    x = ffn(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(output_dim)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
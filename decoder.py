import numpy as np
import tensorflow as tf


class RNN(tf.keras.layers.Layer):

    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.src_vocab_size  = src_vocab_size
        self.tgt_vocab_size  = tgt_vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.encoder =tf.keras.layers.GRU(
            units=self.hidden_size,
            return_sequences=True,
            return_state=True
        )

        self.decoder = tf.keras.layers.GRU(
            units=self.hidden_size,
            return_sequences=True,
            return_state=True
        )

        self.src_embedding = tf.keras.layers.Embedding(
            input_dim=self.src_vocab_size,
            output_dim=self.hidden_size
        )

        self.tgt_embedding = tf.keras.layers.Embedding(
            input_dim=self.tgt_vocab_size,
            output_dim=self.hidden_size
        )

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(units=int(2 * self.tgt_vocab_size), activation='relu'),
            tf.keras.layers.Dense(units=self.tgt_vocab_size)
        ])


    def call(self, src_inputs, tgt_inputs):

        src_embedings = self.src_embedding(src_inputs)

        encoder_output, encoder_state = self.encoder(src_embedings)

        tgt_embedings = self.tgt_embedding(tgt_inputs)

        decoder_output, decoder_state = self.decoder(tgt_embedings, initial_state=encoder_state)

        logits = self.classifier(decoder_output)

        return logits

# class ManualDecoder(nn.Module):
#     def __init__(self, layer, N):
#         super(ManualDecoder, self).__init__()
#
#         #TODO: Initialize the necessary pieces of the decoder
#         # (Hint, the mostly consists of making copies of your decoder layers)
#         self.layers = clones(layer, N)
#         self.norm = self.norm = nn.LayerNorm(layer.size)
#
#     def forward(self, x, memory, src_padding_mask, tgt_mask, tgt_padding_mask):
#         #TODO: Implement the forward pass
#         for layer in self.layers:
#           x = layer(x, memory, src_padding_mask, tgt_mask, tgt_padding_mask)
#         x = self.norm(x)
#         return x

class Transformer(tf.keras.Model):

    def __init__(self,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward = 512,
                 dropout = 0.1):

        super(Transformer, self).__init__()
        self.encoder = ManualEncoder(emb_size, nhead, src_vocab_size, dim_feedforward, dropout)


class ManualEncoder(tf.keras.Model):

    def __init__(self, **kwargs):
        super(ManualEncoder, self).__init__(**kwargs)
        pass


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, size, nhead, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.self_attention = tf.keras.layers.MultiHeadAttention(nhead, size, dropout=dropout)
        self.feed_forward = tf.keras.layers.Dense(size)

        self.norm = tf.keras.layers.LayerNorm(size)
        # self.dropout_fn = tf.keras.layers.Dropout(dropout)
        self.size = size

    def call(self, x, memory, src_padding_mask=None, tgt_mask=None, tgt_padding_mask=None):
        y = self.norm(x)
        y, _ = self.self_attention(y, y, y, tgt_mask)
        x = x + y

        y = self.norm(x)
        ff = self.feed_forward(y)
        x = x + ff

        return x

class ManualDecoder(tf.keras.Model):

    def __init__(self, size, dropout, nhead, **kwargs):
        super(ManualDecoder, self).__init__(**kwargs)
        self.self_attention = tf.keras.layers.MultiHeadAttention(nhead, size, dropout=dropout)
        self.cross_attention = tf.keras.layers.MultiHeadAttention(nhead, size, dropout=dropout)
        self.feed_forward = tf.keras.layers.Dense(size)

        self.norm = tf.keras.layers.LayerNorm(size)
        self.dropout_fn = tf.keras.layers.Dropout(dropout)
        self.size = size

    def call(self, x, memory, src_padding_mask=None, tgt_mask=None, tgt_padding_mask=None):
        y = self.norm(x)
        y, _ = self.self_attention(y, y, y, tgt_mask)
        x = x + y

        y = self.norm(x)
        y, _ = self.cross_attention(y, memory, memory, src_padding_mask)
        x = x + y

        y = self.norm(x)
        ff = self.feed_forward(y)
        x = x + ff

        return x

def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    depth = depth / 2
    ## Generate a range of positions and depths
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    STUDENT MUST WRITE:

    Embed labels and apply positional offsetting
    """

    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size, mask_zero=True)
        self.pos_encoding = positional_encoding(length=window_size, depth=embed_size)

    def call(self, x):
        embdding = self.embedding(x)
        embdding *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        pos_code = self.pos_encoding
        return embdding + pos_code


if __name__ == "__main__":
    pass
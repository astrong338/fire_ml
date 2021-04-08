from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from typeguard import typechecked
import tensorflow as tf

from typing import Union, Optional

# TODO
# 1) Need to only return output during training as in attention model
# 2) Need to clean up tf.cond vs python conditionals


@typechecked
def _create_positional_matrix(max_pos: tf.Tensor, d_model: tf.Tensor):
    positions = tf.range(0, max_pos)
    positions = tf.expand_dims(positions, -1)
    features = tf.range(0, d_model)
    features = tf.expand_dims(features, 0)
    d_model_float = tf.cast(d_model, tf.float32)
    features_num = tf.cast(features // 2, tf.float32)
    angle_rates = 1 / tf.pow(10000, (2 * features_num / d_model_float))
    angle_rads = tf.cast(positions, tf.float32) * angle_rates
    even_rads = tf.transpose(tf.sin(angle_rads[:, 0::2]))
    odd_rads = tf.transpose(tf.cos(angle_rads[:, 1::2]))
    limit = tf.shape(angle_rads)[1]
    even_indices = tf.range(0, limit, delta=2, dtype=tf.int32)
    odd_indices = tf.range(1, limit, delta=2, dtype=tf.int32)
    pos_encoding = tf.dynamic_stitch([even_indices, odd_indices], [even_rads, odd_rads])
    pos_encoding = tf.transpose(pos_encoding)
    pos_encoding = tf.expand_dims(pos_encoding, 0)
    return pos_encoding


class LinearEncOut(Layer):
    """Passes a Encoder Transformer Output through a MLP.

    Used to predict the first value in a auto-regressive encoder-decoder model.

    Examples:

    >>> # TODO

    Args:
        units: The number of hidden units to use.
        num_features: The number of output features to predict.
        activation: The activation function to use after the first linear layer.
        stop_grad: Whether or not to stop the gradients from backpropogating before
            this layer. Default is True.

    Call arguments:
        inputs: a Tensor of rank 3 of the output from a transformer encoder layer.

    Input shapes:
        inputs: (batch_size, input_seq_len, d_model)

    Output shapes:
        output: (batch_size, 1, num_features)

    Returns:
        output: a single step prediction from the encoder output.
    """

    @typechecked
    def __init__(
        self,
        units: tf.Tensor,
        num_features: tf.Tensor,
        activation: str = "relu",
        drop_rate1: Union[float, tf.Tensor] = 0.0,
        drop_rate2: Union[float, tf.Tensor] = 0.0,
        stop_grad: Union[bool, tf.Tensor] = True,
        trainable=True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self._units = units
        self._num_features = (num_features,)
        self._activation = (activation,)
        if isinstance(stop_grad, bool):
            stop_grad = tf.constant(stop_grad)
        self._stop_grad = stop_grad

        self._flat = layers.Flatten()
        self._drop1 = layers.Dropout(drop_rate1)
        self._hidden1 = layers.Dense(units, activation=activation)
        self._drop2 = layers.Dropout(drop_rate2)
        self._pred = layers.Dense(num_features)

    def call(self, inputs, **kwargs):
        x = tf.cond(self._stop_grad, lambda: tf.stop_gradient(inputs), lambda: inputs)
        x = self._flat(x)
        x = self._drop1(x)
        x = self._hidden1(x)
        x = self._drop2(x)
        output = self._pred(x)
        return output


class ScaledDotProductAttention(Layer):
    """Computes scaled dot product attention.

    Masking is used to remove unwanted attention connections. Padding can help with
    variable length input or output sequences and with ensuring that outputs can only
    pay attention to previous outputs.

    Examples:

    >>> # batch_size is 3, 5 queries, query/key depth is 8, 15 keys/values,
    >>> # value depth is 19
    >>> query_shape = (3, 5, 8)
    >>> key_shape = (3, 15, 8)
    >>> value_shape = (3, 15, 19)
    >>> queries = tf.random.normal(query_shape)
    >>> keys = tf.random.normal(key_shape)
    >>> values = tf.random.normal(key_shape)
    >>> output, attention_weights = _scaled_dot_product_attention(
            queries,
            keys,
            values
        )
    >>> print(output.shape)
    (3, 5, 19)
    >>> print(attention_weights.shape)
    (3, 5, 15)

    Args:
        queries: The set of queries to compare with the keys.
        keys: The keys assocatied with each value that are used to calculate the
            compatibility with the query.
        values: The values used in calculating the output as a weighted sum of
            the values based on the compatibility.
        mask: Masks query-key connnections to prevent illegal connections. In an
            Encoder-Decoder where the output of one step in the decoder is fed
            back into the model to predict the next step, transformers would allowing
            "peaking" into the future without masking. Default is None (no masking).

    Input shapes:
        queries: (batch_size, # queries, query_key_depth).
        keys: (batch_size, # keys/values, query_key_depth).
        values: (batch_size, # keys/values, value_depth).
        mask: broadcastable to (batch_size, # queries, # keys / values)

    Output shapes:
        output: (batch_size, # queries, value_depth)
        attention_weights: (batch_size, # queries, # keys/values)

    Returns:
        output: The weighted sum of the values for each query
        attention_weights: The attention weights for each query-key comparison.
    """

    @typechecked
    def __init__(
        self,
        trainable: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super(ScaledDotProductAttention, self).__init__(
            trainable=trainable, name=name, **kwargs
        )

    def call(
        self,
        queries: tf.Tensor,
        keys: tf.Tensor,
        values: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ):
        # matmul_qk shape == (batch_size, # queries, # keys/values)
        matmul_qk = tf.matmul(queries, keys, transpose_b=True)
        # scale matmul_qk to adjust for large dot products at higher depths.
        # this keeps the softmax from blowing up.
        dk = tf.cast(tf.shape(keys)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (
                mask * -1e9
            )  # large negative values make softmax zero
        # axis=-1 causes the axis to normalize on the last dimension (keys)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, values)
        return output, attention_weights


@typechecked
def _scaled_dot_product_attention(
    queries: tf.Tensor,
    keys: tf.Tensor,
    values: tf.Tensor,
    mask: Optional[tf.Tensor] = None,
):
    """Computes scaled dot product attention.

    Masking is used to remove unwanted attention connections. Padding can help with
    variable length input or output sequences and with ensuring that outputs can only
    pay attention to previous outputs.

    Examples:

    >>> # batch_size is 3, 5 queries, query/key depth is 8, 15 keys/values,
    >>> # value depth is 19
    >>> query_shape = (3, 5, 8)
    >>> key_shape = (3, 15, 8)
    >>> value_shape = (3, 15, 19)
    >>> queries = tf.random.normal(query_shape)
    >>> keys = tf.random.normal(key_shape)
    >>> values = tf.random.normal(key_shape)
    >>> output, attention_weights = _scaled_dot_product_attention(
            queries,
            keys,
            values
        )
    >>> print(output.shape)
    (3, 5, 19)
    >>> print(attention_weights.shape)
    (3, 5, 15)

    Args:
        queries: The set of queries to compare with the keys.
        keys: The keys assocatied with each value that are used to calculate the
            compatibility with the query.
        values: The values used in calculating the output as a weighted sum of
            the values based on the compatibility.
        mask: Masks query-key connnections to prevent illegal connections. In an
            Encoder-Decoder where the output of one step in the decoder is fed
            back into the model to predict the next step, transformers would allowing
            "peaking" into the future without masking. Default is None (no masking).

    Input shapes:
        queries: (batch_size, # queries, query_key_depth).
        keys: (batch_size, # keys/values, query_key_depth).
        values: (batch_size, # keys/values, value_depth).
        mask: broadcastable to (batch_size, # queries, # keys / values)

    Output shapes:
        output: (batch_size, # queries, value_depth)
        attention_weights: (batch_size, # queries, # keys/values)

    Returns:
        output: The weighted sum of the values for each query
        attention_weights: The attention weights for each query-key comparison.
    """
    # matmul_qk shape == (batch_size, # queries, # keys/values)
    matmul_qk = tf.matmul(queries, keys, transpose_b=True)
    # scale matmul_qk to adjust for large dot products at higher depths.
    # this keeps the softmax from blowing up.
    dk = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (
            mask * -1e9
        )  # large negative values make softmax zero
    # axis=-1 causes the axis to normalize on the last dimension (keys)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, values)
    return output, attention_weights


class MultiHeadAttention(Layer):
    """Preforms multi-headed attention.

    Examples:

    >>> # TODO

    Arguments:
        d_model: The total number of hidden units in Q, K, and V
        num_heads: The number of times d_model is split up.
        attention_function: A Callable that is used to calculate the attention output
            and weights. Must take queries, keys, values, and mask as inputs. Default
            is scaled dot product attention.

    Call arguments:
        queries: The set of queries to compare with the keys.
        keys: The keys assocatied with each value that are used to calculate the
            compatibility with the query.
        values: The values used in calculating the output as a weighted sum of
            the values based on the compatibility.
        mask: Masks query-key connnections to prevent illegal connections. In an
            Encoder-Decoder where the output of one step in the decoder is fed
            back into the model to predict the next step, transformers would allowing
            "peaking" into the future without masking. Default is None (no masking).

    Input shapes:
        queries: (batch_size, # queries, query_key_depth).
        keys: (batch_size, # keys/values, query_key_depth).
        values: (batch_size, # keys/values, value_depth).
        mask: broadcastable to (batch_size, # queries, # keys / values)

    Output shapes:
        output: (batch_size, # queries, d_model)
        attention_weights: (batch_size, num_heads, # queries, # keys/values)

    Returns:
        output: The weighted sum of the values for each query
        attention_weights: The attention weights for each query-key comparison.
    """

    @typechecked
    def __init__(
        self,
        d_model: tf.Tensor,
        num_heads: tf.Tensor,
        attention_function=ScaledDotProductAttention,
        trainable: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__(
            trainable=trainable, name=name, **kwargs
        )
        self._d_model = d_model
        self._depth = d_model // num_heads
        self._num_heads = num_heads
        self._attention_function = attention_function(**kwargs)
        self._validate_init()

        # Define Layers
        # This makes the queries, keys, and values all have the same depth.
        # Technically, the keys could have a different depth from the queires and keys.
        self._wq = layers.Dense(d_model)
        self._wk = layers.Dense(d_model)
        self._wv = layers.Dense(d_model)

        self._dense = layers.Dense(d_model)

    def _validate_init(self):
        # make sure that the total dimension is divisible by the number of heads.
        if tf.math.floormod(self._d_model, self._num_heads) != tf.constant(0):
            raise ValueError("Depth must be divisible but the number of heads.")

    def _split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).

        Transpose the result so that the shape is (batch_size, num_heads, |T|, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self._num_heads, self._depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(
        self, queries: tf.Tensor, keys: tf.Tensor, values: tf.Tensor, mask: tf.Tensor
    ):
        batch_size = tf.shape(queries)[0]
        q = self._wq(queries)
        k = self._wk(keys)
        v = self._wv(values)

        # split queries, keys, and values into multiple heads
        q = self._split_heads(
            q, batch_size
        )  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split_heads(
            k, batch_size
        )  # (batch_size, num_heads, seq_len_k_v, depth)
        v = self._split_heads(
            v, batch_size
        )  # (batch_size, num_heads, seq_len_k_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, |Q|, depth)
        # attention_weights.shape == (batch_size, num_heads, |Q|, |K|)
        scaled_attention, attention_weights = self._attention_function(
            q, k, v, mask
        )  # (batch_size, |Q|, num_heads, d_model // num_heads)
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self._d_model)
        )  # (batch_size, |Q|, d_model)
        output = self._dense(concat_attention)  # (batch_size, |Q|, d_model)

        return output, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self._d_model,
                "num_heads": self._num_heads,
                "attention_function": self._attention_function,
            }
        )
        return config


class TransEncUnit(Layer):
    """A standard Transformer Encoder layer.

    Examples:

    >>> # TODO

    Arguments:
        d_model: The total number of hidden units in Q, K, and V
        num_heads: The number of times d_model is split up.
        feedf_units: The number of units in the feed forward network,
        attention_function: A Callable that is used to calculate the attention output
            and weights. Must take queries, keys, values, and mask as inputs. Default
            is scaled dot product attention.
        drop_rate: The frequency to set inputs to the layer norms to zero during
            training.

    Call arguments:
        x: An input sequence.
        return_attention_weights: sets whether or not to return the attention weights.
        mask: Masks query-key connnections to prevent illegal connections. In an
            Encoder-Decoder where the output of one step in the decoder is fed
            back into the model to predict the next step, transformers would allowing
            "peaking" into the future without masking. Default is None (no masking).

    Input shapes:
        x: (batch_size, |T_in|, # features) where |T_in| is the number of time steps
            in the input sequence. NOTE d_model must equal # features for the skip
            steps to work. Otherwise, the code will raise an error of incompatible
            shapes.
        mask: broadcastable to (batch_size, |T_in|, |T_in|)

    Output shapes:
        output: (batch_size, |T_in|, d_model)
        attention_weights: (batch_size, num_heads, |T_in|, |T_in|)

    Returns:
        enc_unit_out: The self-attention encoding of the input.
        attn_w: The attention weights for each self-attention comparison.
    """

    def __init__(
        self,
        d_model: tf.Tensor,
        num_heads: tf.Tensor,
        feedf_units: tf.Tensor,
        attention_function=ScaledDotProductAttention,
        drop_rate: tf.Tensor = tf.constant(0.1),
        trainable=True,
        name=None,
        **kwargs,
    ):
        super(TransEncUnit, self).__init__(trainable=trainable, name=name, **kwargs)
        self._d_model = d_model
        self._num_heads = num_heads
        self._feedf_units = feedf_units
        self._attention_function = attention_function
        self._drop_rate = drop_rate

        self._mha = MultiHeadAttention(d_model, num_heads, attention_function, **kwargs)
        self._ff1 = layers.Dense(feedf_units, activation="relu")
        self._ff2 = layers.Dense(d_model)
        self._layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self._layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self._dropout1 = layers.Dropout(drop_rate)
        self._dropout2 = layers.Dropout(drop_rate)

    def call(
        self,
        x: tf.Tensor,
        training: Union[bool, tf.Tensor, None] = None,
        mask: Optional[tf.Tensor] = None,
    ):
        attn_out, attn_w = self._mha(x, x, x, mask)
        attn_out = self._dropout1(attn_out, training=training)
        out1 = self._layernorm1(x + attn_out)

        ffn_out = self._ff1(out1)
        ffn_out = self._ff2(ffn_out)
        enc_unit_out = self._layernorm2(out1 + ffn_out)

        return enc_unit_out, attn_w

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self._d_model,
                "num_heads": self._num_heads,
                "feedf_units": self._feedf_units,
                "attention_function": self._attention_function,
                "drop_rate": self._drop_rate,
            }
        )
        return config


class TransDecUnit(Layer):
    """A single standard Transformer Decoder layer.

    Examples:

    >>> # TODO

    Arguments:
        d_model: The total number of hidden units in Q, K, and V
        num_heads: The number of times d_model is split up.
        feedf_units: The number of units in the feed forward network,
        attention_function: A Callable that is used to calculate the attention output
            and weights. Must take queries, keys, values, and mask as inputs. Default
            is scaled dot product attention.
        drop_rate: The frequency to set inputs to the layer norms to zero during
            training.

    Call arguments:
        x: An input sequence.
        return_attention_weights: sets whether or not to return the attention weights.
        mask: Masks query-key connnections to prevent illegal connections. In an
            Encoder-Decoder where the output of one step in the decoder is fed
            back into the model to predict the next step, transformers would allowing
            "peaking" into the future without masking. Default is None (no masking).

    Input shapes:
        x: (batch_size, |T_in|, # features) where |T_in| is the number of time steps
            in the input sequence. NOTE d_model must equal # features for the skip
            steps to work. Otherwise, the code will raise an error of incompatible
            shapes.
        mask: broadcastable to (batch_size, |T_in|, |T_in|)

    Output shapes:
        output: (batch_size, |T_in|, d_model)
        attention_weights: (batch_size, num_heads, |T_in|, |T_in|)

    Returns:
        enc_unit_out: The self-attention encoding of the input.
        attn_w: The attention weights for each self-attention comparison.
    """

    def __init__(
        self,
        d_model: tf.Tensor,
        num_heads: tf.Tensor,
        feedf_units: tf.Tensor,
        attention_function=ScaledDotProductAttention,
        drop_rate: tf.Tensor = tf.constant(0.1),
        trainable=True,
        name=None,
        **kwargs,
    ):
        super(TransDecUnit, self).__init__(trainable=trainable, name=name, **kwargs)
        self._d_model = d_model
        self._num_heads = num_heads
        self._feedf_units = feedf_units
        self._attention_function = attention_function
        self._drop_rate = drop_rate

        self._mha1 = MultiHeadAttention(
            d_model,
            num_heads,
            attention_function=attention_function,
            trainable=trainable,
            **kwargs,
        )
        self._mha2 = MultiHeadAttention(
            d_model,
            num_heads,
            attention_function=attention_function,
            trainable=trainable,
            **kwargs,
        )
        self._ff1 = layers.Dense(feedf_units, activation="relu")
        self._ff2 = layers.Dense(d_model)

        self._layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self._layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self._layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self._drop1 = layers.Dropout(drop_rate)
        self._drop2 = layers.Dropout(drop_rate)
        self._drop3 = layers.Dropout(drop_rate)

    def call(
        self,
        x: tf.Tensor,
        enc_output: tf.Tensor,
        training: Union[bool, tf.Tensor, None] = None,
        look_ahead_mask: Optional[tf.Tensor] = None,
        padding_mask: Optional[tf.Tensor] = None,
        **kwargs,
    ):
        attn1, attn_w_1 = self._mha1(x, x, x, look_ahead_mask)
        attn1 = self._drop1(attn1, training=training)
        out1 = self._layernorm1(attn1 + x)

        attn2, attn_w_2 = self._mha2(out1, enc_output, enc_output, mask=padding_mask)
        attn2 = self._drop2(attn2, training=training)
        out2 = self._layernorm2(attn2 + out1)

        ffn_out = self._ff1(out2)
        ffn_out = self._ff2(ffn_out)
        ffn_out = self._drop3(ffn_out, training=training)
        out3 = self._layernorm3(ffn_out + out2)

        return out3, attn_w_1, attn_w_2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self._d_model,
                "num_heads": self._num_heads,
                "feedf_units": self._feedf_units,
                "attention_function": self._attention_function,
                "drop_rate": self._drop_rate,
            }
        )
        return config


class TransEncoder(Layer):
    """A stack of transformer encoder layers.

    Examples:

    >>> # TODO

    Arguments:
        d_model: The total number of hidden units in Q, K, and V
        num_heads: The number of times d_model is split up.
        feedf_units: The number of units in the feed forward network,
        attention_function: A Callable that is used to calculate the attention output
            and weights. Must take queries, keys, values, and mask as inputs. Default
            is scaled dot product attention.
        drop_rate: The frequency to set inputs to the layer norms to zero during
            training.

    Call arguments:
        x: An input sequence.
        return_attention_weights: sets whether or not to return the attention weights.
        mask: Masks query-key connnections to prevent illegal connections. In an
            Encoder-Decoder where the output of one step in the decoder is fed
            back into the model to predict the next step, transformers would allowing
            "peaking" into the future without masking. Default is None (no masking).

    Input shapes:
        x: (batch_size, |T_in|, # features) where |T_in| is the number of time steps
            in the input sequence. NOTE d_model must equal # features for the skip
            steps to work. Otherwise, the code will raise an error of incompatible
            shapes.
        mask: broadcastable to (batch_size, |T_in|, |T_in|)

    Output shapes:
        output: (batch_size, |T_in|, d_model)
        attention_weights: (batch_size, num_heads, |T_in|, |T_in|)

    Returns:
        enc_unit_out: The self-attention encoding of the input.
        attn_w: The attention weights for each self-attention comparison.
    """

    def __init__(
        self,
        num_layers: tf.Tensor,
        d_model: tf.Tensor,
        num_heads: tf.Tensor,
        feedf_units: tf.Tensor,
        attention_function=ScaledDotProductAttention,
        drop_rate: tf.Tensor = tf.constant(0.1),
        absolute_encoding: Union[bool, tf.Tensor] = False,
        trainable: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self._d_model = d_model
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._feedf_units = feedf_units
        self._attention_function = attention_function
        self._drop_rate = drop_rate
        self._absolute_encoding = absolute_encoding
        self._pos_encoding = None

        self._embedding = layers.Dense(d_model)

        self._enc_layers = [
            TransEncUnit(
                d_model,
                num_heads,
                feedf_units,
                attention_function=attention_function,
                drop_rate=drop_rate,
            )
            for _ in tf.range(num_layers)
        ]

        self._drop = layers.Dropout(drop_rate)

    def build(self, input_shape):
        input_seq_len = input_shape[1]
        if isinstance(input_seq_len, int):
            input_seq_len = tf.constant(input_seq_len)
        self._input_seq_len = input_seq_len
        self._num_output_features = tf.constant(input_shape[-1])
        if self._absolute_encoding:
            self._pos_encoding = _create_positional_matrix(input_seq_len, self._d_model)

    def call(
        self,
        x: tf.Tensor,
        training: Union[bool, tf.Tensor, None] = None,
        mask: Optional[tf.Tensor] = None,
        **kwargs,
    ):
        pos_encoding = self._pos_encoding
        x = self._embedding(x)
        x = tf.cond(self._absolute_encoding, lambda: x + pos_encoding, lambda: x)
        x = self._drop(x, training=training)

        attn_ws = tf.TensorArray(tf.float32, self._num_layers)
        i = tf.constant(0)
        inc = tf.constant(1)
        for enc_layer in self._enc_layers:
            x, attn_w = enc_layer(
                x,
                training=training,
                mask=mask,
            )
            attn_ws = attn_ws.write(i, attn_w)
            i += inc

        attn_ws = tf.transpose(attn_ws.stack(), [1, 0, 2, 3, 4])

        return x, attn_ws


class TransDecoder(Layer):
    """A stack of standard Transformer Decoder layers.

    Examples:

    >>> # TODO

    Arguments:
        d_model: The total number of hidden units in Q, K, and V
        num_heads: The number of times d_model is split up.
        feedf_units: The number of units in the feed forward network,
        attention_function: A Callable that is used to calculate the attention output
            and weights. Must take queries, keys, values, and mask as inputs. Default
            is scaled dot product attention.
        drop_rate: The frequency to set inputs to the layer norms to zero during
            training.

    Call arguments:
        x: An input sequence.
        return_attention_weights: sets whether or not to return the attention weights.
        mask: Masks query-key connnections to prevent illegal connections. In an
            Encoder-Decoder where the output of one step in the decoder is fed
            back into the model to predict the next step, transformers would allowing
            "peaking" into the future without masking. Default is None (no masking).

    Input shapes:
        x: (batch_size, |T_in|, # features) where |T_in| is the number of time steps
            in the input sequence. NOTE d_model must equal # features for the skip
            steps to work. Otherwise, the code will raise an error of incompatible
            shapes.
        mask: broadcastable to (batch_size, |T_in|, |T_in|)

    Output shapes:
        output: (batch_size, |T_in|, d_model)
        attention_weights: (batch_size, num_heads, |T_in|, |T_in|)

    Returns:
        enc_unit_out: The self-attention encoding of the input.
        attn_w: The attention weights for each self-attention comparison.
    """

    @typechecked
    def __init__(
        self,
        prediction_steps: tf.Tensor,
        num_layers: tf.Tensor,
        d_model: tf.Tensor,
        num_heads: tf.Tensor,
        feedf_units: tf.Tensor,
        attention_function=ScaledDotProductAttention,
        drop_rate: tf.Tensor = tf.constant(0.1),
        absolute_encoding: Union[bool, tf.Tensor] = False,
        trainable: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self._prediction_steps = prediction_steps
        self._num_output_features = None
        self._num_layers = num_layers
        self._d_model = d_model
        self._num_heads = num_heads
        self._feedf_units = feedf_units
        self._attention_function = attention_function
        self._drop_rate = drop_rate
        self._absolute_encoding = absolute_encoding

        self._embedding = layers.Dense(d_model)
        if absolute_encoding:
            self._pos_encoding = _create_positional_matrix(prediction_steps, d_model)
        else:
            self._pos_encoding = None

        self._dec_layers = [
            TransDecUnit(
                d_model,
                num_heads,
                feedf_units,
                attention_function=attention_function,
                drop_rate=drop_rate,
                trainable=trainable,
            )
            for _ in tf.range(num_layers)
        ]
        self._drop = layers.Dropout(drop_rate)

    def build(self, input_shape):
        self._num_output_features = input_shape[-1]
        self._out_dense = layers.Dense(self._num_output_features)

    def call(
        self,
        x,
        enc_output,
        training: Optional[bool] = None,
        look_ahead_mask: Optional[tf.Tensor] = None,
        padding_mask: Optional[tf.Tensor] = None,
        **kwargs,
    ):
        x_shape = tf.shape(x)
        out_steps = x_shape[1]
        pos_encoding = self._pos_encoding[:, :out_steps, :]
        x = self._embedding(x)
        # ERROR is here
        x = tf.cond(self._absolute_encoding, lambda: x + pos_encoding, lambda: x)
        x = self._drop(x, training=training)

        attn_ws_1 = tf.TensorArray(tf.float32, self._num_layers)
        attn_ws_2 = tf.TensorArray(tf.float32, self._num_layers)
        i = tf.constant(0)
        inc = tf.constant(1)
        for dec_layer in self._dec_layers:
            x, attn_w_1, attn_w_2 = dec_layer(
                x,
                enc_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
            )
            attn_ws_1 = attn_ws_1.write(i, attn_w_1)
            attn_ws_2 = attn_ws_2.write(i, attn_w_2)
            i += inc
        output = self._out_dense(x)
        attn_ws_1 = tf.transpose(attn_ws_1.stack(), [1, 0, 2, 3, 4])
        attn_ws_2 = tf.transpose(attn_ws_2.stack(), [1, 0, 2, 3, 4])
        return output, attn_ws_1, attn_ws_2

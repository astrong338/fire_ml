from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from typeguard import typechecked
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf

from typing import Union, Optional


def _validate_units(units: int):
    """Ensures that units are greater than 0.

    Args:
        units: the number of hidden units for a layer.

    Raises:
        ValueError: exception if the number of hidden units <= 0.
    """
    if units <= 0:
        raise ValueError(
            (
                "The number of hidden units must be greater than 0." "Received units={}"
            ).format(units)
        )


@keras_export("keras.layers.RnnEncoder")
class RnnEncoder(Layer):
    """Encodes time series input with a LSTM or GRU into tensors of hidden states.

    This layer creates a an encoder layer that uses an LSTM or GRU to encode a time
    series input into a hidden state for each time step. The hidden state from each
    time step is passed forward for the calculation of the next.

    When using this layer as the first layer in a model, provide an input_shape
    argument.

    NOTE: an LSTM encoder will return two tensors for final state instead of
    a single tensor. There is one tensor for the final hidden state and another
    for the final cell state. A GRU does not have a cell state like an LSTM does.

    Examples:

    >>> # The inputs are variable bactch size and 128 time steps and 5 features.
    >>> # rnn_type is "LSTM"
    >>> input_shape = (512, 128, 5)
    >>> x = tf.random.normal(input_shape)
    >>> y = PI_ML.layers.RnnEncoder(32, rnn_type="LSTM", input_shape=input_shape)
    >>> print(y[0].shape)
    >>> print(y[1].shape)
    >>> print(y[2].shape)
    (512, 128, 32)
    (512, 32)
    (512, 32)

    >>> # The inputs are variable bactch size and 128 time steps and 5 features.
    >>> # rnn_type is "GRU"
    >>> input_shape = (512, 128, 5)
    >>> x = tf.random.normal(input_shape)
    >>> y = PI_ML.layers.RnnEncoder(32, rnn_type="GRU", input_shape=input_shape)
    >>> print(y[0].shape)
    (512, 128, 32)
    >>> print(y[1].shape)
    (512, 32)

    Arguments:
        units: the number of hidden units in the RNN.
        rnn_type: LSTM or GRU. This will select the type of RNN used for
            encoding. The default value is LSTM.

    Call arguments:
        inputs: a Tensor of rank 3 that holds the input for the encoder.

    Input shape: The shape is (None, |T|, # features) where |T| is the number of time
            steps in the input. The first dimension is None because it represents
            the batch size which can vary.

    Output shapes:
        output: (None, |T|, # hidden units).
        hidden_state: (None, # hidden units).
        cell_state (LSTM only): (None, # hidden units)

    Returns:
        output: a Tensor of rank 3 of the hidden states at each step of the RNN.
        hidden_state: a Tensor of rank 2 of the final hidden state of the RNN.
            NOTE: hidden_state == output[:, -1, :]
        cell_state: a Tensor of rank 2 of the final cell state of the RNN.

    Raises:
        ValueError: when units <= 0.
    """

    @typechecked
    def __init__(
        self,
        units: int,
        rnn_type: str = "LSTM",
        trainable: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super(RnnEncoder, self).__init__(trainable=trainable, name=name, **kwargs)
        if isinstance(units, float):
            units = int(round(units))
        self._units = units
        self._rnn_type = rnn_type
        _validate_units(units)
        self._rnn: Union[layers.RNN, layers.GRU, None] = None
        self._is_LSTM = False
        if rnn_type == "LSTM":
            self._is_LSTM = True
            self._rnn_cell = layers.LSTMCell(
                self._units, recurrent_initializer="glorot_uniform"
            )
            self._rnn = layers.RNN(
                self._rnn_cell, return_sequences=True, return_state=True
            )
        else:
            self._rnn = layers.GRU(
                self._units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            )

    def call(self, inputs, training=None):
        """Returns a tensor of the hidden units and final hidden state."""
        if self._is_LSTM:
            output, hidden_state, cell_state = self._rnn(inputs, training=training)
            return output, hidden_state, cell_state
        else:
            output, hidden_state = self._rnn(inputs, training=training)
            return output, hidden_state

    def get_config(self):
        config = super().get_config()
        config.update({"units": self._units, "rnn_type": self._rnn_type})
        return config


@keras_export("keras.layers.BahdanauAttention")
class BahdanauAttention(Layer):
    """Implements a basic attention mechanism as described in Bahdanaue et. al.

    In general, the query is the previous hidden state of the decoder. The keys
    and values are equal to each other with both being the hidden states at
    each step of the RNN. The scores (or energy) are a measure of the alignment
    between the previous hidden state of the decoder (query) and the hidden states
    of the encoder RNN (keys). In this class, additive style attention is used to
    score the match between the query and each key.

    The attention weights are a softmax of the query-key pairs where the sum in the
    softmax is across the steps of the RNN. This makes it so that the sum
    of the attention weights is 1.

    The context vector is the weighted sum of the attention weights and values where
    the values are the hidden states at each step of the encoder.

    Examples:

    >>> # Batch size: 32, enc. hidden units: 64, attention hidden units: 8,
    >>> # dec. hidden units: 16, 50 time steps
    >>> query_shape = (32, 16)
    >>> values_shape = (32, 50, 64)
    >>> query = tf.random.normal(query_shape)
    >>> values = tf.random.normal(values_shape)
    >>> attention_layer = BahdanauAttention(8)
    >>> context_vector, attention_weights = attention_layer(query, values)
    >>> print(f"Context shape: {context_vector.shape}")
    (32, 64)
    >>> print(f"Attention Weights shape: {attention_weights.shape}")
    (32, 50)

    Arguments:
        units: the number of hidden units to use in the feed forward network
            that is used to calculate the score between the query and keys.

    Call arguments:
        query: The query tensor used to compare to the keys. The shape is:
            (None, # dec. hidden units). The first dimension is None to represent
            the number of batches which can be variable.
        values: A tensor that serves as the keys for calculating the attention
            weights. It also provides the values in the weighted sum between the
            attention weights and values to calculate the context vector. The shape
            of values is (None, |T|, # enc. hidden units) where |T| is the number
            of steps in the RNN.

    Input shapes:
        query: (None, # dec. hidden units).
        values: (None, |T|, # enc. hidden units).

    Output shapes:
        context_vector: (None, # enc. hidden units).
        attention_weights: (None, |T|).

    Returns:
        context_vector: The weighted sum of the encoder hidden states.
            Shape is (None, # enc. hidden units).
        attention_weights: The attention weights used to create the context vector.
            Shape is (None, |T|).

    Raises:
        ValueError: when units <= 0.
    """

    @typechecked
    def __init__(
        self,
        units: int,
        name: Optional[str] = None,
        trainable: bool = True,
        **kwargs,
    ):
        super(BahdanauAttention, self).__init__(
            trainable=trainable, name=name, **kwargs
        )
        _validate_units(units)
        self._units = units
        # NOTE all of the following are linear operations.
        # By default, the activation is None.
        self._W1 = layers.Dense(units)  # calculates dot(inputs, W1) + bias
        self._W2 = layers.Dense(units)
        self._V = layers.Dense(1)

    def call(self, query, values):
        """When called, calculates the context vector and attention weights."""
        # First we must add a time dimension to the query. This gives query and value
        # the same number of dimensions.

        # Query and value must have the same number of dimensions so that dot(W1, query)
        # can be broadcast to the shape of dot(W2, values).

        # The same query value will be added to each time step in values.
        query_with_time_axis = tf.expand_dims(
            query, 1
        )  # Shape is now (None, 1, # dec. hidden units)
        score_additive_part = self._W1(query_with_time_axis) + self._W2(
            values
        )  # Addition of query and keys. Shape: (None, |T|, units)

        # Applies non-linear activation to the additive product and maps it
        # to a single value. Shape: (None, |T|, 1)
        score = self._V(tf.nn.tanh(score_additive_part))

        # Takes a softmax of the score between the query and each key over
        # the sum of scores.
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = (
            attention_weights * values
        )  # Shape is (None, |T|, # enc. hidden units)

        # remove the last dimension of the attention weights which has size 1
        # Shape is (None, |T|, 1)
        attention_weights = tf.squeeze(attention_weights, [2])
        # Shape is (None, |T|)

        context_vector = tf.reduce_sum(
            context_vector, axis=1
        )  # Shape is (None, # enc. hidden units)
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self._units})
        return config


@keras_export("keras.layers.RnnDecoder")
class RnnDecoder(Layer):
    """Implements a RNN based decoder with or without attention.

    Examples:

    >>> # batch size: 8, encoder units: 9, decoder units: 18, RNN Type: LSTM,
    >>> # Use attention is True, attention units is 15, number of time steps in the
    >>> # encoder is 30, the number of features in the decoder is 32
    >>> decoder = RnnDecoder(18,
                             rnn_type="LSTM",
                             use_attention=True,
                             attention_units=15)
    >>> input_size = (8, 32)
    >>> hidden_size = (8, 18)
    >>> enc_state_size = (8, 30, 9)
    >>> cell_size = (8, 18)
    >>> x = tf.random.normal(input_size)
    >>> prev_hidden = tf.random.normal(hidden_size)
    >>> enc_hidden_states = tf.random.normal(enc_state_size)
    >>> prev_cell_state = tf.random.normal(cell_size)
    >>> y_pred, hidden_state, cell_state, attention_weights = decoder(
            x, prev_hidden, enc_hidden_states, prev_cell_state
        )
    >>> print(y_pred.shape)
    (8, 18)
    >>> print(hidden_state.shape)
    (8, 18)
    >>> print(cell_state.shape)
    (8, 18)
    >>> print(attention_weights.shape)
    (8, 30)

    Arguments:
        units: The number of hidden units to use for the decoder.
            Must be greater than 0.
        rnn_type: Either "LSTM" or "GRU".
        use_attention: Whether the decoder should use attention or not. Default is True.
        attention_units: The number of units to use for the feed forward network used
            in calculating the attention scores. Default is None. If None is given
            and attention is used, the number of units will default to the number of
            units used in the decoder.

    Call arguments:
        x: The input used at the current time step.
        prev_hidden: The hidden state of the either the previous step of the encoder or
            decoder. The first step of the decoder should use the final hidden state of
            the encoder. After that, the model should use the previous hidden state of
            the decoder.
        enc_hidden_states: The hidden states from each step of the encoder.
        prev_cell_state: The cell states from the previous step of the encoder or
            decoder. The first step of the decoder should use the final cell state of
            the encoder if the encoder is a LSTM. After that, the model should use
            the previous cell state of the decoder.

    Input shapes:
        x: (None, # features).
        prev_hidden: (None, # dec. units)
        enc_hidden_states: (None, |T|, # enc. units)
        prev_cell_state: (None, # dec. units)

    Output shapes:
        y_pred: (None, # dec. units)
        state: (None, # dec. units)
        attention_weights: (None, |T|) where |T| is the number of time steps
            in the encoder.

    Returns:
        y_pred and state: the output hidden (and cell for LSTM) state of the cell.
        attention_weights: The attention weights created for a single step of the
            decoder.
    """

    @typechecked
    def __init__(
        self,
        units: int,
        rnn_type: str = "LSTM",
        use_attention: bool = True,
        attention_units: int = None,
        trainable: bool = True,
        name: str = "decoder",
        **kwargs,
    ):
        super(RnnDecoder, self).__init__(trainable=trainable, name=name, **kwargs)
        _validate_units(units)
        self._units = units
        self._rnn_type = rnn_type
        self._use_attention = use_attention
        self._rnn_cell: Union[layers.LSTMCell, layers.GRUCell, None] = None
        self._double_state = False
        if rnn_type == "LSTM":
            self._double_state = True
            self._rnn_cell = layers.LSTMCell(units)
        else:
            self._rnn_cell = layers.GRUCell(units)
        self._attention: Union[BahdanauAttention, None] = None
        if use_attention:
            if attention_units is None or not isinstance(attention_units, int):
                attention_units = units
            self._attention = BahdanauAttention(attention_units)
        self._attention_units = attention_units

    def call(
        self,
        x: tf.Tensor,
        prev_hidden: tf.Tensor,
        enc_hidden_states: tf.Tensor,
        prev_cell_state: Optional[tf.Tensor] = None,
        training: Optional[bool] = None,
    ):
        use_attention = self._use_attention
        if use_attention:
            context_vector, attention_weights = self._attention(
                prev_hidden, enc_hidden_states
            )
            x = tf.concat([context_vector, x], axis=-1)
        else:
            attention_weights = tf.constant([])
        if self._double_state:
            prev_states = [prev_hidden, prev_cell_state]
            y_pred, state = self._rnn_cell(x, states=prev_states, training=training)
            hidden_state = state[0]
            cell_state = state[1]
            return y_pred, hidden_state, cell_state, attention_weights
        else:
            y_pred, hidden_state = self._rnn_cell(
                x, states=prev_hidden, training=training
            )
            return y_pred, hidden_state, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self._units,
                "rnn_type": self._rnn_type,
                "use_attention": self._use_attention,
                "attention_units": self._attention_units,
            }
        )
        return config

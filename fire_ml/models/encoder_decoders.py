from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow import keras

from tensorflow.keras import Model
import tensorflow as tf

from typeguard import typechecked
from typing import Union, Optional

from fire_ml.layers.rnn_encoder_decoder import RnnEncoder, RnnDecoder


class AttentionEncoderDecoder(Model):
    """Encoder-Decoder model with attention.

    This model is a classic Encoder-Decoder style model with Bahdanau style attention
    and uses either LSTMs or GRUs for the encoder and decoders.

    At the moment, either only LSTMs or only GRUs are used for both encoder and decoder.
    However, the layers it depends on do allow for mix and match of the two. This is
    something that could be tried at a future date.

    Examples:

    >>> # batch size: 8, encoder units: 9, decoder units: 18, RNN Type: LSTM,
    >>> # Use attention is True, attention units is 15, number of time steps in the
    >>> # encoder is 30, the number of features in the decoder is 3, the number of
    >>> # prediction steps is 20
    >>> model = AttentionEncoderDecoder(
            9,
            18,
            20,
            3,
            attention_units = 15,
            rnn_type = "LSTM",
            drop_rate = 0.5,
            idx_continue = [1]
        )
    >>> input_size = (8, 32, )
    >>> input = tf.random.normal(input_size)
    >>> predictions, attention_weights = model(input, return_attention_weights=True)
    >>> print(predictions.shape)
    (8, 20, 3)
    >>> print(attention_weights.shape)
    (8, 20, 30)

    Arguments:
        enc_units: the number of hidden units to use for the encoder.
        dec_units: the number of hidden units to use for the decoder.
        prediction_steps: the number of time steps to predict into the future.
        num_output_features: the number of output features to predict at each time step.
            This needs to match the number of features in the labels provided for
            training.
        attention_units: the number of hidden units to use for attention scoring.
            If None, it will use the same number of units as Encoder and Decoder layers.
            The default is None.
        rnn_type: Selects the type of RNN. Choices are "LSTM" or "GRU". Default is LSTM.
        drop_rate: the rate to use in dropout. Dropout is applied to the hidden units
            returned by each step of the decoder before the dense layer predicts the
            output features.
        idx_continue: A list of indices from input series that is in common with the
            prediction series. The order of the indices needs to match the order of the
            features in the prediction series.
        input_min_max: a list, numpy array, or tensor of the min and max values for the
            input series. If this is not None, the model will do a min/max
            normalization. Default is None.
        output_min_max: a tuple of a list, numpy array, or tensor of the min and max
            values for the prediction series. If this is not None, the model will do a
            min/max normalization of the ground truth predicted series during training
            and it will preform the inverse of the min/max normalization when the model
            is not training. Default is None.
        trainable_layers: a list of the trainable layers. Add 0 or "encoder" to make the
            encoder layer trainable. Add 1 or "decoder" to make the decoder layer
            trainable. The default is None. If it's None, both encoder and decoder are
            trianable.

        Call arguments:
            inputs: a Tensor of rank 3 that holds the input for the encoder.
            training: whether or not the model is training. When the model is not
                training, dropout is disabled. Default is None.
            ground_truths: if training is true and this is not None, it will use teacher
                forcing. Default is None.
            return_attention_weights: If this is True, calling the model will return the
                attention weights in addition to the predictions. Default is False.

        Input shape: The shape is (None, |T|, # features) where |T| is the number of
            time steps in the input. The first dimension is None because it represents
            the batch size which can vary.

        Output shape:
            y_pred: (None, prediction_steps, num_output_features).
            attention_weights: (None, prediction_steps, input steps)

        Returns:
            y_pred: A tensor predicting prediction steps into the future for the
                selected output features.
            attention_weights (optional): the attention weights generated while
                executing the model.

    """

    @typechecked
    def __init__(
        self,
        enc_units: int,
        dec_units: int,
        prediction_steps: int,
        num_output_features: int,
        attention_units: Optional[int] = None,
        rnn_type: str = "LSTM",
        drop_rate: float = 0.5,
        idx_continue: Optional[list] = None,
        input_min_max: Optional[tuple] = None,
        output_min_max: Optional[tuple] = None,
        trainable_layers: Optional[list] = None,
        **kwargs,
    ):
        super(AttentionEncoderDecoder, self).__init__(**kwargs)
        # initialize all parameters.
        # this allows us to write the get_config method.
        self._enc_units = enc_units
        self._dec_units = dec_units
        self._prediction_steps = prediction_steps
        self._num_output_features = num_output_features
        self._attention_units = attention_units
        if attention_units is None:
            self._attention_units = self._dec_units
        self._rnn_type = rnn_type
        self._isLSTM = False
        if rnn_type == "LSTM":
            self._isLSTM = True
        self._drop_rate = drop_rate
        self._idx_continue = idx_continue
        if idx_continue is None:
            self._continue_encoder_end = False
        else:
            self._continue_encoder_end = True
        self._input_min_max = input_min_max
        self._output_min_max = output_min_max
        self._trainable_layers = trainable_layers

        self._validate_init()

        # set scaling factors
        input_min: Union[tf.Tensor, None] = None
        input_max: Union[tf.Tensor, None] = None
        output_min: Union[tf.Tensor, None] = None
        output_max: Union[tf.Tensor, None] = None
        self._scale_inputs = False
        self._scale_outputs = False
        if input_min_max is not None:
            input_min = tf.constant(input_min_max[0], dtype=tf.float32)
            input_max = tf.constant(input_min_max[1], dtype=tf.float32)
            self._scale_inputs = True
        if output_min_max is not None:
            output_min = tf.constant(output_min_max[0], tf.float32)
            output_max = tf.constant(output_min_max[1], tf.float32)
            self._scale_outputs = True
        self._input_min, self._input_max = input_min, input_max
        self._output_min, self._output_max = output_min, output_max

        # initialize layers
        train_encoder = False
        train_decoder = False
        if trainable_layers is None:
            train_encoder = True
            train_decoder = True
        else:
            if 0 in trainable_layers or "encoder" in trainable_layers:
                train_encoder = True
            if 1 in trainable_layers or "decoder" in trainable_layers:
                train_decoder = True
        self._encoder = RnnEncoder(enc_units, rnn_type, trainable=train_encoder)
        self._decoder = RnnDecoder(
            dec_units,
            rnn_type=rnn_type,
            use_attention=True,
            attention_units=attention_units,
            trainable=train_decoder,
        )
        self._dropout = keras.layers.Dropout(drop_rate)
        self._layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._fc = tf.keras.layers.Dense(num_output_features)

    def _validate_init(self):
        if self._enc_units < 1:
            raise ValueError("Encoder units must be greater than 0.")
        if self._dec_units < 1:
            raise ValueError("Decoder units must be greater than 0.")
        if self._attention_units < 1:
            raise ValueError("Attention units must be greater than 0.")
        if self._idx_continue is not None:
            if len(self._idx_continue) > self._num_output_features:
                raise ValueError("Too many indices to select from the input features.")

    def _get_initial_hidden(
        self, hidden_state: tf.Tensor, cell_state: Optional[tf.Tensor] = None
    ):
        enc_units = self._enc_units
        dec_units = self._dec_units
        if enc_units == dec_units:
            dec_hidden, dec_cell = hidden_state, cell_state
        else:
            enc_hidden_shape = tf.shape(hidden_state)
            batch_size = enc_hidden_shape[0]
            dec_hidden = tf.zeros((batch_size, dec_units))
            if cell_state is None:
                dec_cell = None
            else:
                dec_cell = tf.zeros((batch_size, dec_units))
        return dec_hidden, dec_cell

    def call(
        self,
        inputs,
        training: Optional[bool] = None,
        ground_truths: Optional[tf.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        """
        inputs: a Tensor of rank 3 that holds the input for the encoder.
        training: whether or not the model is training. When the model is not
        training, dropout is disabled. Default is None.
        ground_truths: if training is true and this is not None, it will use teacher
            forcing. Default is None.
        return_attention_weights: If this is True, calling the model will return the
            attention weights in addition to the predictions. Default is False.
        """
        # 1. Set scaling for inputs
        if self._scale_inputs:
            input_min = self._input_min
            input_max = self._input_max
            inputs = (inputs - input_min) / (input_max - input_min)

        # 2. Get encoder output
        if self._isLSTM:
            enc_hidden_states, prev_hidden, prev_cell_state = self._encoder(inputs)
        else:
            enc_hidden_states, prev_hidden = self._encoder(inputs)
            prev_cell_state = None

        # 3. determine the first input for the decoder layer
        x_shape = tf.shape(inputs)  # this is the dynamic shape evaluated at run time.
        batch_size = x_shape[0]
        idx_continue = self._idx_continue
        if idx_continue is not None:
            x_last_step = inputs[:, -1, :]
            x_continue = tf.gather(x_last_step, idx_continue, axis=1)
            num_continued_features = len(idx_continue)
            pad_amount = self._num_output_features - num_continued_features
            if pad_amount > 0:
                padding = tf.convert_to_tensor(
                    [[0, 0], [0, pad_amount]], dtype=tf.int32
                )
                dec_input = tf.pad(x_continue, padding)
            else:
                dec_input = x_continue
        else:
            dec_input = tf.zeros((batch_size, self._num_output_features))

        # 4. Make sure the hidden and cell states are the correct size for the decoder
        prev_hidden, prev_cell_state = self._get_initial_hidden(
            prev_hidden, prev_cell_state
        )

        # 4. Autoregressive calculation of sequential time steps
        if len(dec_input.shape) < 2:
            dec_input = tf.expand_dims(dec_input, -1)

        total_predictions = self._prediction_steps
        predictions = tf.TensorArray(tf.float32, size=total_predictions)
        attention_weights = tf.TensorArray(tf.float32, size=total_predictions)
        for i in range(total_predictions):
            if self._isLSTM:
                y_pred, hidden_state, cell_state, attention_weight = self._decoder(
                    dec_input,
                    prev_hidden,
                    enc_hidden_states,
                    prev_cell_state=prev_cell_state,
                    training=training,
                )
            else:
                y_pred, hidden_state, attention_weight = self._decoder(
                    dec_input, prev_hidden, enc_hidden_states, training=training
                )
                cell_state = None
            y_pred = self._dropout(y_pred, training=training)
            y_pred = self._layer_norm(y_pred)
            y_pred = self._fc(y_pred)
            predictions = predictions.write(i, y_pred)
            if return_attention_weights:
                attention_weights = attention_weights.write(i, attention_weight)
            if ground_truths is not None:
                dec_input = ground_truths[:, i, :]
            else:
                dec_input = y_pred
            prev_hidden = hidden_state
            prev_cell_state = cell_state
        predictions = predictions.stack()
        predictions = tf.transpose(predictions, [1, 0, 2])  # put batches first
        predict_shape = tf.shape(predictions)
        predictions = tf.cond(
            tf.equal(predict_shape[-1], tf.constant(1)),
            lambda: tf.squeeze(predictions, axis=-1),
            lambda: predictions,
        )
        output_min = self._output_min
        if output_min is not None:
            output_max = self._output_max
            predictions = predictions * (output_max - output_min) + output_min
        if return_attention_weights is False:
            return predictions

        attention_weights = attention_weights.stack()
        print(attention_weights.shape)
        attention_weights = tf.transpose(attention_weights, [1, 0, 2])
        return predictions, attention_weights

    def get_config(self):
        config = super().get_config()
        """
        enc_units: int,
        dec_units: int,
        prediction_steps: int,
        num_output_features: int,
        attention_units: Optional[int] = None,
        rnn_type: str = "LSTM",
        drop_rate: float = 0.5,
        idx_continue: Optional[list] = None,
        input_min_max: Optional[tuple] = None,
        output_min_max: Optional[tuple] = None,
        trainable_layers: Optional[list] = None,
        """
        config.update(
            {
                "enc_units": self._enc_units,
                "dec_units": self._dec_units,
                "prediction_steps": self._prediction_steps,
                "num_output_features": self._num_output_features,
                "attention_units": self._attention_units,
                "rnn_type": self._rnn_type,
                "drop_rate": self._drop_rate,
                "idx_continue": self._idx_continue,
                "input_min_max": self._input_min_max,
                "output_min_max": self._output_min_max,
                "trainable_layers": self._trainable_layers,
            }
        )

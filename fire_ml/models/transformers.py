from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import Model, layers
import tensorflow as tf
from tensorflow.python.keras.layers.core import Activation

from typeguard import typechecked
from typing import Union, Optional

from fire_ml.layers.transformer import LinearEncOut, TransEncoder, TransDecoder

# TODO
# 1) Need to only return output during training as in attention model
# 2) Need to clean up tf.cond vs python conditionals


class ClassicTransformer(Model):
    """The original Transformer adapted to time series prediction.

    Args:
        prediction_steps: the number of time steps to predict.
        num_output_features: the number of features in each prediction step.
        num_layers: the number of layers to use for the encoder and decoder.
        d_model: the depth of the model.
        num_heads: the number of heads to break the model depth into. d_model must be
            divisible by num_heads.
        ff_units: the number of hidden units to use in the feed forward network of the
            transformer layers.
        drop_rate: the the dropout rate to use between each step of the transformer
            layers.
        first_prediciton_units: If using a FF network to predict the first step in the
            output, this gives the number of hidden units to use.
        idx_continue: the indices from the input to use in the first step of the
            decoder. These need to be less than the number of output features
            and will appear as the first indices in the decoder output. Make
            sure that this matches up with the data.
    """

    @typechecked
    def __init__(
        self,
        prediction_steps: tf.Tensor,
        num_output_features: Union[int, tf.Tensor],
        num_layers: tf.Tensor,
        d_model: tf.Tensor,
        num_heads: tf.Tensor,
        ff_units: tf.Tensor,
        drop_rate: tf.Tensor,
        first_prediction_units: Union[int, tf.Tensor, None] = None,
        idx_continue: Optional[list] = None,
        input_min_max: Optional[tuple] = None,
        output_min_max: Optional[tuple] = None,
        stop_grad: bool = False,
        *args,
        **kwargs,
    ):
        super(ClassicTransformer, self).__init__(*args, **kwargs)
        self._prediction_steps = prediction_steps
        if isinstance(first_prediction_units, int):
            first_prediction_units = tf.constant(first_prediction_units)
        self._first_prediction_units = first_prediction_units
        self._num_layers = num_layers
        self._d_model = d_model
        self._num_heads = num_heads
        self._ff_units = ff_units
        self._drop_rate = drop_rate
        self._input_min_max = input_min_max
        self._output_min_max = output_min_max
        self._stop_grad = stop_grad
        self._idx_continue = idx_continue
        if idx_continue is None:
            self._continue_encoder_end = False
        else:
            self._continue_encoder_end = True
        if not self._continue_encoder_end and first_prediction_units is None:
            self._dec_inital_zeros = True
        else:
            self._dec_inital_zeros = False

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

        if isinstance(num_output_features, int):
            num_output_features = tf.constant(num_output_features)
        self._num_output_features = num_output_features

        self._encoder = TransEncoder(
            num_layers,
            d_model,
            num_heads,
            ff_units,
            drop_rate=drop_rate,
            absolute_encoding=tf.constant(True),
        )
        if not self._continue_encoder_end and not self._dec_inital_zeros:
            self._predict_first = True
            self._first_pred = LinearEncOut(
                first_prediction_units,
                num_output_features,
                activation="relu",
                drop_rate1=0.5,
                drop_rate2=0.1,
            )
        else:
            self._predict_first = False
            self._first_pred = None
        self._predict_first_tensor = tf.constant(self._predict_first)

        self._final_layer = layers.Dense(num_output_features)

        prediction_steps = tf.constant(prediction_steps)
        self._decoder = TransDecoder(
            prediction_steps,
            num_layers,
            d_model,
            num_heads,
            ff_units,
            drop_rate=drop_rate,
            absolute_encoding=tf.constant(True),
        )

    def _create_look_ahead_mask(self, output: tf.Tensor):
        size = tf.shape(output)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def call(
        self,
        inputs,
        target: Optional[tf.Tensor] = None,
        training: Optional[bool] = None,
        return_attention_weights: Union[bool, tf.Tensor] = False,
        enc_padding_mask: Optional[tf.Tensor] = None,
        dec_padding_mask: Optional[tf.Tensor] = None,
    ):
        if self._scale_inputs:
            input_min = self._input_min
            input_max = self._input_max
            inputs = (inputs - input_min) / (input_max - input_min)
        enc_output, enc_attn = self._encoder(
            inputs,
            training=training,
            mask=enc_padding_mask,
            return_attention_weights=return_attention_weights,
        )
        if self._predict_first:
            y_pred_1 = self._first_pred(enc_output)
        elif self._continue_encoder_end:
            idx_continue = self._idx_continue
            x_continue = tf.gather(inputs[:, -1, :], idx_continue, axis=-1)
            num_continued_features = len(idx_continue)
            pad_amount = self._num_output_features - num_continued_features
            if pad_amount > 0:
                padding = tf.convert_to_tensor(
                    [[0, 0], [0, pad_amount]], dtype=tf.int32
                )
                y_pred_1 = tf.pad(x_continue, padding)
            else:
                y_pred_1 = x_continue
        else:
            inputs_shape = tf.shape(inputs)
            batch_size = inputs_shape[0]
            y_pred_1 = tf.zeros((batch_size, self._num_output_features))

        tf.ensure_shape(y_pred_1, [None, self._num_output_features])

        if target is None:
            # only need to predict the second and after time steps.
            if not self._predict_first:
                pred_steps = self._prediction_steps
            else:
                pred_steps = self._prediction_steps - tf.constant(1)
            output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            output = output.write(0, y_pred_1)
            for i in tf.range(pred_steps):
                out_inter = tf.transpose(output.stack(), [1, 0, 2])
                look_ahead_mask = self._create_look_ahead_mask(out_inter)
                # y_preds, dec_attn1, dec_attn2 = self._decoder(
                y_preds, _, _ = self._decoder(
                    out_inter,
                    enc_output,
                    training=training,
                    look_ahead_mask=look_ahead_mask,
                    padding_mask=dec_padding_mask,
                    return_attention_weights=return_attention_weights,
                )
                y_pred = y_preds[:, -1, :]
                output = output.write(i, y_pred)
            output = tf.transpose(output.stack(), [1, 0, 2])
        else:
            output_min = self._output_min
            if output_min is not None:
                output_max = self._output_max
                target = (target - output_min) / (output_max - output_min)
            look_ahead_mask = self._create_look_ahead_mask(target)
            # output, dec_attn1, dec_attn2 = self._decoder(
            output, _, _ = self._decoder(
                target,
                enc_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=dec_padding_mask,
                return_attention_weights=return_attention_weights,
            )
            if self._predict_first:
                output = tf.concat([y_pred_1, output[:, 1:, :]], axis=1)
            else:
                output = tf.concat([y_pred_1, output], axis=1)
        output = tf.cond(
            self._predict_first_tensor, lambda: output, lambda: output[:, 1:, :]
        )
        output_min = self._output_min
        if output_min is not None:
            output_max = self._output_max
            output = output * (output_max - output_min) + output_min
        if tf.shape(output)[-1] == 1:
            output = tf.squeeze(output, axis=-1)
        return output
        """
        if not return_attention_weights or training:
            return output
        else:
            return output, enc_attn, dec_attn1, dec_attn2
        """


class TransformerClassifier(Model):
    """Self-Attention for time series classification.

    Args:
        num_classifications: the number of classifications to make.
        num_layers: the number of layers to use for the encoder.
        d_model: the depth of the model.
        num_heads: the number of heads to break the model depth into. d_model must be
            divisible by num_heads.
        ff_units: the number of hidden units to use in the feed forward network of the
            transformer layers.
        drop_rate: the the dropout rate to use between each step of the transformer
            layers.

    Call arguments:
        inputs: a Tensor representing a multivariate time series
        training: whether the model is training or not.
        return_attention_weights: sets whether or not to return the attention weights.

    Input shapes:
        inputs: (batch_size, num_time_steps, # features).

    Output shape:
        output: (batch_size, num_classifications).
        attn_wts: (batch_size, num_layers, num_heads, num_time_steps, num_time_steps)

    Returns:
        output: classifications (0 or 1). Multiple classifications allows to classify
            at multiple time horizons. For example, it could classify whether a plug
            loss will happen in the next 5 mins, the next 10 mins, the next 15 mins, etc.
        attn_w: The attention weights for each self-attention comparison.
    """

    @typechecked
    def __init__(
        self,
        num_classifications: tf.Tensor,
        num_layers: tf.Tensor,
        d_model: tf.Tensor,
        num_heads: tf.Tensor,
        ff_units: tf.Tensor,
        drop_rate: tf.Tensor = tf.constant(0.1),
        activation: Union[str, Activation, None] = "sigmoid",
        **kwargs,
    ):
        super(TransformerClassifier, self).__init__(**kwargs)
        self._num_classifications = num_classifications
        self._num_layers = num_layers
        self._d_model = d_model
        self._num_heads = num_heads
        self._ff_units = ff_units
        self._drop_rate = drop_rate

        self._encoder = TransEncoder(
            num_layers,
            d_model,
            num_heads,
            ff_units,
            drop_rate=drop_rate,
            absolute_encoding=tf.constant(True),
        )
        self._flatten = layers.Flatten()
        self._dense = layers.Dense(num_classifications, activation=activation)

    def call(
        self,
        inputs: tf.Tensor,
        training: Union[bool, tf.Tensor, None] = None,
        return_attention_weights: bool = False,
    ):
        enc_output, enc_attn = self._encoder(
            inputs,
            training=training,
        )
        flat_encodings = self._flatten(enc_output)
        output = self._dense(flat_encodings)

        if return_attention_weights is False:
            return output

        return output, enc_attn

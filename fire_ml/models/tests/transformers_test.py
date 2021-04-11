from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire_ml.models.transformers import TransformerClassifier

import tensorflow as tf


def test_TransformerClassifier():
    input_shape = (32, 12, 4)
    num_class = tf.constant(1)
    num_layers = tf.constant(6)
    d_model = tf.constant(256)
    num_heads = tf.constant(4)
    ff_units = tf.constant(256)

    trans_class = TransformerClassifier(
        num_class,
        num_layers,
        d_model,
        num_heads,
        ff_units,
    )
    inputs = tf.random.uniform(input_shape)
    classification, attn = trans_class(inputs, return_attention_weights=True)

    assert classification.shape == (32)
    assert attn.shape == (32, 6, 4, 12, 12)

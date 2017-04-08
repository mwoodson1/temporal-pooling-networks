import tensorflow as tf
import numpy as np

from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import math_ops, array_ops

def linear(args, output_size, bias, bias_start=None, scope=None):
    """
    This is a slightly modified version of _linear used by Tensorflow rnn.
    The only change is that we have allowed bias_start=None.
    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        elif bias_start is None:
            bias_term = tf.get_variable("Bias", [output_size], dtype=dtype)
        else:
            bias_term = tf.get_variable("Bias", [output_size], dtype=dtype,
                                        initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return res + bias_term

RNNCell = core_rnn_cell.RNNCell

class HighwayRNNCell(RNNCell):
    """Highway RNN Network with multiplicative_integration"""

    def __init__(self, num_units, num_highway_layers=3, use_inputs_on_each_layer=False):
        self._num_units = num_units
        self.num_highway_layers = num_highway_layers
        self.use_inputs_on_each_layer = use_inputs_on_each_layer


    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, timestep=0, scope=None):
        current_state = state
        for highway_layer in xrange(self.num_highway_layers):
            with tf.variable_scope('highway_factor_'+str(highway_layer)):
                if self.use_inputs_on_each_layer or highway_layer == 0:
                    highway_factor = tf.tanh(linear([inputs, current_state], self._num_units, True))
                else:
                    highway_factor = tf.tanh(linear([current_state], self._num_units, True))
            with tf.variable_scope('gate_for_highway_factor_'+str(highway_layer)):
                if self.use_inputs_on_each_layer or highway_layer == 0:
                    gate_for_highway_factor = tf.sigmoid(linear([inputs, current_state],
                                                         self._num_units, True, -3.0))
                else:
                    gate_for_highway_factor = tf.sigmoid(linear([current_state],
                                                         self._num_units, True, -3.0))

                gate_for_hidden_factor = 1.0 - gate_for_highway_factor

            current_state = highway_factor * gate_for_highway_factor + current_state * gate_for_hidden_factor

        return current_state, current_state

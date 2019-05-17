import tensorflow as tf
from tensorflow.contrib import cudnn_rnn


def rnn(rnn_type, inputs, hidden_size,training, layer_num=1, dropout_keep_prob=None):
    cell = get_cell(rnn_type, hidden_size, layer_num, 'bidirectional')
    inputs = tf.transpose(inputs, [1, 0, 2])
    outputs, state = cell(inputs, training=training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, state


def get_cell(rnn_type, hidden_size, layer_num=1, direction='bidirectional'):
    if rnn_type.endswith('lstm'):
        cudnn_cell = cudnn_rnn.CudnnLSTM(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                         dropout=0)
    elif rnn_type.endswith('gru'):
        cudnn_cell = cudnn_rnn.CudnnGRU(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                        dropout=0)
    elif rnn_type.endswith('rnn'):
        cudnn_cell = cudnn_rnn.CudnnRNNTanh(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                            dropout=0)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    return cudnn_cell
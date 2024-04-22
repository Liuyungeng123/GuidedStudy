import tensorflow as tf


class GAT(tf.keras.layers.Layer):
    """
    Graph Attention Layer according to https://arxiv.org/pdf/1710.10903.pdf
    """
    def __init__(self, output_dim, use_bias, activation, num_heads=4, reduction='concat', kernel_regularizer=None, **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)

        self.num_heads = num_heads
        self.reduction = reduction
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (self.num_heads, input_dim, self.output_dim)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        self.attn_self_weight = self.add_weight(shape=(self.num_heads, self.output_dim, 1),
                                                initializer='glorot_uniform',
                                                name='attn_self',
                                                regularizer=self.kernel_regularizer,
                                                trainable=True)
        self.attn_neigh_weight = self.add_weight(shape=(self.num_heads, self.output_dim, 1),
                                                 initializer='glorot_uniform',
                                                 name='attn_neighbors',
                                                 regularizer=self.kernel_regularizer,
                                                 trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_heads, self.output_dim,),
                                        initializer='zeros',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def call(self, inputs):
        features = [tf.keras.backend.dot(inputs[0], self.kernel[k]) for k in range(self.num_heads)]
        attn_self_output = [tf.keras.backend.dot(features[k], self.attn_self_weight[k]) for k in range(self.num_heads)]
        attn_neigh_output = [tf.keras.backend.dot(features[k], self.attn_neigh_weight[k]) for k in range(self.num_heads)]
        dense = [tf.nn.leaky_relu(attn_self_output[k] + tf.transpose(attn_neigh_output[k], [0, 2, 1]), alpha=0.2) for k in range(self.num_heads)]
        mask = -10e9 * (1.0 - inputs[1])
        dense = [dense[k] + mask for k in range(self.num_heads)]

        self.normalized_att_scores = [tf.nn.softmax(dense[k], axis=-1) for k in range(self.num_heads)]
        output = [tf.keras.backend.batch_dot(self.normalized_att_scores[k], features[k]) for k in range(self.num_heads)]

        if self.use_bias:
            output = [tf.keras.backend.bias_add(output[k], self.bias[k]) for k in range(self.num_heads)]

        if self.reduction == 'concat':
            output = tf.keras.backend.concatenate(output, axis=-1)
        else:
            output = tf.keras.layers.Average()(output)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'num_heads': self.num_heads,
            'reduction': self.reduction,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class GraphConv(tf.keras.layers.Layer):
    """
         Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    """

    def __init__(self, output_dim, use_bias, activation, kernel_regularizer=None, **kwargs):
        super(GraphConv, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def _normalize(self, A, eps=1e-6):
        n = tf.shape(A)[-1]
        A -= tf.linalg.diag(tf.linalg.diag_part(A))
        A_hat = A + tf.cast(tf.eye(n), dtype=A.dtype)[tf.newaxis, :, :]
        D_hat = tf.linalg.diag(1./(eps + tf.math.sqrt(tf.reduce_sum(A_hat, axis=2))))
        return tf.matmul(tf.matmul(D_hat, A_hat), D_hat)

    def call(self, inputs):
        output = tf.keras.backend.batch_dot(self._normalize(inputs[1]), inputs[0])
        output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class FuncPredictor(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(FuncPredictor, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.output_layer = tf.keras.layers.Dense(2*output_dim)
        self.reshape = tf.keras.layers.Reshape(target_shape=(output_dim, 2))
        self.softmax = tf.keras.layers.Softmax(axis=-1, name='labels')

    def call(self, x):
        x = self.output_layer(x)
        x = self.reshape(x)
        out = self.softmax(x)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config


class SumPooling(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(SumPooling, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        x_pool = tf.reduce_sum(x, axis=self.axis)
        return x_pool

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis,
        })
        return config


class SAGPooling(tf.keras.layers.Layer):
    def __init__(self, ratio, **kwargs):
        super(SAGPooling, self).__init__(**kwargs)
        self.ratio = ratio

    def call(self, x, scores):
        num_nodes = tf.shape(x)[0]
        num_nodes_to_keep = tf.cast(num_nodes * self.ratio, tf.int32)

        # 根据分数进行排序并选取前 num_nodes_to_keep 个节点
        _, indices = tf.math.top_k(scores, k=num_nodes_to_keep)
        x_pool = tf.gather(x, indices)

        return x_pool

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'ratio': self.ratio,
        })
        return config

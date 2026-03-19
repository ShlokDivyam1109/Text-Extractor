import numpy as np
from Neural_Network import Functions 

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=0, stride=1,
                 activation_function='ReLu', learning_rate=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.stride = stride
        self.learning_rate = learning_rate
        self.F = Functions
        fan_in = in_channels * kernel_size * kernel_size
        self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros(out_channels)
        self.activation_function = activation_function

    def pad_input(self, input):
        pad_h, pad_w = self.padding
        if pad_h == 0 and pad_w == 0:
            return input
        padded = np.pad(
            input,
            pad_width=((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )
        return padded

    def forward(self, input):
        self.input_original = input.copy()
        padded_input = self.pad_input(input)
        self.input_padded = padded_input
        H, W, C = padded_input.shape
        K = self.kernel_size
        S = self.stride
        out_h = (H - K) // S + 1
        out_w = (W - K) // S + 1
        output = np.zeros((out_h, out_w, self.out_channels))
        for f in range(self.out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * S
                    w_start = j * S
                    for c in range(self.in_channels):
                        for kh in range(K):
                            for kw in range(K):
                                output[i, j, f] += padded_input[h_start + kh, w_start + kw, c] * self.filters[f, c, kh, kw]
                    output[i, j, f] += self.biases[f]
        self.pre_activation = output.copy()
        if self.activation_function == 'ReLu':
            output = np.maximum(0, output)
        elif self.activation_function == 'Sigmoid':
            output = 1 / (1 + np.exp(-output))
        self.output = output
        return output

    def backward(self, d_output):
        if self.activation_function == 'ReLu':
            d_pre = d_output * (self.pre_activation > 0).astype(float)
        elif self.activation_function == 'Sigmoid':
            d_pre = d_output * self.output * (1 - self.output)
        else:
            d_pre = d_output
        d_filters = np.zeros_like(self.filters)
        d_biases = np.sum(d_pre, axis=(0, 1))
        H_orig, W_orig, _ = self.input_original.shape
        pad_h, pad_w = self.padding
        S = self.stride
        out_h, out_w, _ = d_pre.shape
        d_padded = np.zeros_like(self.input_padded)
        for f in range(self.out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * S
                    w_start = j * S
                    for c in range(self.in_channels):
                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                val = self.input_padded[h_start + kh, w_start + kw, c]
                                d_filters[f, c, kh, kw] += d_pre[i, j, f] * val
                                d_padded[h_start + kh, w_start + kw, c] += d_pre[i, j, f] * self.filters[f, c, kh, kw]
        d_input = d_padded[pad_h : pad_h + H_orig, pad_w : pad_w + W_orig, :]
        self.filters -= self.learning_rate * d_filters
        self.biases -= self.learning_rate * d_biases
        return d_input

import numpy as np

class PoolingLayer:
    def __init__(self, pool_size=2, stride=None, pool_type='max'):
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = (stride, stride) if stride is not None else self.pool_size
        self.pool_type = pool_type.lower()
        if self.pool_type not in ['max', 'avg']:
            raise ValueError("pool_type must be 'max' or 'avg'")
        self.input_shape = None
        self.out_shape = None
        self.max_positions = None

    def forward(self, input):
        self.input = input.copy()
        self.input_shape = input.shape
        H, W, C = input.shape
        ph, pw = self.pool_size
        sh, sw = self.stride
        out_h = (H - ph) // sh + 1
        out_w = (W - pw) // sw + 1
        self.out_shape = (out_h, out_w, C)
        output = np.zeros((out_h, out_w, C))
        if self.pool_type == 'max':
            self.max_positions = np.zeros((out_h, out_w, C, 2), dtype=int)
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * sh
                    w_start = j * sw
                    patch = input[h_start : h_start + ph,w_start : w_start + pw,c]
                    if self.pool_type == 'max':
                        max_val = np.max(patch)
                        output[i, j, c] = max_val
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        self.max_positions[i, j, c] = [h_start + max_idx[0],w_start + max_idx[1]]
                    else:
                        output[i, j, c] = np.mean(patch)
        return output

    def backward(self, d_output):
        d_input = np.zeros_like(self.input)
        out_h, out_w, C = d_output.shape
        ph, pw = self.pool_size
        sh, sw = self.stride
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * sh
                    w_start = j * sw
                    if self.pool_type == 'max':
                        pos_h, pos_w = self.max_positions[i, j, c]
                        d_input[pos_h, pos_w, c] += d_output[i, j, c]
                    else:
                        patch_area = ph * pw
                        grad_patch = d_output[i, j, c] / patch_area
                        d_input[h_start : h_start + ph,
                                w_start : w_start + pw,
                                c] += grad_patch
        return d_input

class CNN:
    def __init__(self, config, learning_rate=0.01):
        """
        config example: [('C', 5), ('M', 2), ('C', 10), ('M', 2)]
        'C' = Convolution layer with that many filters (output channels)
        'M' = MaxPooling layer with that pool size
        """
        self.layers = []
        current_channels = 1
        for layer_type, param in config:
            if layer_type == 'C':
                layer = ConvLayer(current_channels, param,kernel_size=3,activation_function='ReLu',learning_rate=learning_rate)
                self.layers.append(layer)
                current_channels = param
            elif layer_type == 'M':
                layer = PoolingLayer(pool_size=param)
                self.layers.append(layer)
        self.learning_rate = learning_rate

    def forward(self, input_image):
        x = input_image
        for layer in self.layers:
            x = layer.forward(x)
        self.feature_shape = x.shape
        flattened = x.flatten()
        return flattened

    def back_propagation(self, d_flattened):
        d_x = d_flattened.reshape(self.feature_shape)
        for layer in reversed(self.layers):
            d_x = layer.backward(d_x)

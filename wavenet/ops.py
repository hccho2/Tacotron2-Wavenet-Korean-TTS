#  coding: utf-8
import tensorflow as tf
import numpy as np
def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}
def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)  # tf.log1p(x) = log(1+x)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels, quantization=True):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        if quantization:
            signal = 2 * (tf.to_float(output) / mu) - 1
        else:
            signal = output
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude


class SubPixelConvolution(tf.layers.Conv2D):
    '''Sub-Pixel Convolutions are vanilla convolutions followed by Periodic Shuffle.

    They serve the purpose of upsampling (like deconvolutions) but are faster and less prone to checkerboard artifact with the right initialization.
    In contrast to ResizeConvolutions, SubPixel have the same computation speed (when using same n° of params), but a larger receptive fields as they operate on low resolution.
    '''
    def __init__(self, filters, kernel_size, padding, strides, NN_init, NN_scaler, up_layers, name=None, **kwargs):
        #Output channels = filters * H_upsample * W_upsample
        conv_filters = filters * strides[0] * strides[1]

        #Create initial kernel
        self.NN_init = NN_init
        self.up_layers = up_layers
        self.NN_scaler = NN_scaler
        init_kernel = tf.constant_initializer(self._init_kernel(kernel_size, strides, conv_filters), dtype=tf.float32) if NN_init else None

        #Build convolution component and save Shuffle parameters.
        super(SubPixelConvolution, self).__init__(
            filters=conv_filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding=padding,
            kernel_initializer=init_kernel,
            bias_initializer=tf.zeros_initializer(),
            data_format='channels_last',
            name=name, **kwargs)

        self.out_filters = filters
        self.shuffle_strides = strides
        self.scope = 'SubPixelConvolution' if None else name

    def build(self, input_shape):
        '''Build SubPixel initial weights (ICNR: avoid checkerboard artifacts).

        To ensure checkerboard free SubPixel Conv, initial weights must make the subpixel conv equivalent to conv->NN resize.
        To do that, we replace initial kernel with the special kernel W_n == W_0 for all n <= out_channels.
        In other words, we want our initial kernel to extract feature maps then apply Nearest neighbor upsampling.
        NN upsampling is guaranteed to happen when we force all our output channels to be equal (neighbor pixels are duplicated).
        We can think of this as limiting our initial subpixel conv to a low resolution conv (1 channel) followed by a duplication (made by PS).

        Ref: https://arxiv.org/pdf/1707.02937.pdf
        '''
        #Initialize layer
        super(SubPixelConvolution, self).build(input_shape)

        if not self.NN_init:
            #If no NN init is used, ensure all channel-wise parameters are equal.
            self.built = False

            #Get W_0 which is the first filter of the first output channels
            W_0 = tf.expand_dims(self.kernel[:, :, :, 0], axis=3) #[H_k, W_k, in_c, 1]

            #Tile W_0 across all output channels and replace original kernel
            self.kernel = tf.tile(W_0, [1, 1, 1, self.filters]) #[H_k, W_k, in_c, out_c]

        self.built = True

    def call(self, inputs):
        with tf.variable_scope(self.scope) as scope:
            #Inputs are supposed [batch_size, freq, time_steps, channels]
            convolved = super(SubPixelConvolution, self).call(inputs)

            #[batch_size, up_freq, up_time_steps, channels]
            return self.PS(convolved)

    def PS(self, inputs):
        #Get different shapes
        #[batch_size, H, W, C(out_c * r1 * r2)]
        batch_size = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]   #W = tf.shape(inputs)[2]
        C = inputs.shape[-1]
        r1, r2 = self.shuffle_strides #supposing strides = (freq_stride, time_stride)
        out_c = self.out_filters #number of filters as output of the convolution (usually 1 for this model)

        assert C == r1 * r2 * out_c

        #Split and shuffle (output) channels separately. (Split-Concat block)
        Xc = tf.split(inputs, out_c, axis=3) # out_c x [batch_size, H, W, C/out_c]
        outputs = tf.concat([self._phase_shift(x, batch_size, H, W, r1, r2) for x in Xc], 3) #[batch_size, r1 * H, r2 * W, out_c]

        with tf.control_dependencies([tf.assert_equal(out_c, tf.shape(outputs)[-1]),
            tf.assert_equal(H * r1, tf.shape(outputs)[1])]):
            outputs = tf.identity(outputs, name='SubPixelConv_output_check')

        return tf.reshape(outputs, [tf.shape(outputs)[0], r1 * H, tf.shape(outputs)[2], out_c])

    def _phase_shift(self, inputs, batch_size, H, W, r1, r2):
        #Do a periodic shuffle on each output channel separately
        x = tf.reshape(inputs, [batch_size, H, W, r1, r2]) #[batch_size, H, W, r1, r2]

        #Width dim shuffle
        x = tf.transpose(x, [4, 2, 3, 1, 0]) #[r2, W, r1, H, batch_size]
        x = tf.batch_to_space_nd(x, [r2], [[0, 0]]) #[1, r2*W, r1, H, batch_size]
        x = tf.squeeze(x, [0]) #[r2*W, r1, H, batch_size]

        #Height dim shuffle
        x = tf.transpose(x, [1, 2, 0, 3]) #[r1, H, r2*W, batch_size]
        x = tf.batch_to_space_nd(x, [r1], [[0, 0]]) #[1, r1*H, r2*W, batch_size]
        x = tf.transpose(x, [3, 1, 2, 0]) #[batch_size, r1*H, r2*W, 1]

        return x

    def _init_kernel(self, kernel_size, strides, filters):
        '''Nearest Neighbor Upsample (Checkerboard free) init kernel size
        '''
        overlap = kernel_size[1] // strides[1]
        init_kernel = np.zeros(kernel_size, dtype=np.float32)
        i = kernel_size[1] // 2
        j = [kernel_size[0] // 2 - 1, kernel_size[0] // 2] if kernel_size[0] % 2 == 0 else [kernel_size[0] // 2]
        for j_i in j:
            init_kernel[j_i,i] = 1. / max(overlap, 1.) if kernel_size[1] % 2 == 0 else 1.

        init_kernel = np.tile(np.expand_dims(init_kernel, 2), [1, 1, 1, filters])

        return init_kernel * (self.NN_scaler)**(1/self.up_layers)

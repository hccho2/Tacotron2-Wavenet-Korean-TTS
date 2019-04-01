#  coding: utf-8
import numpy as np
import tensorflow as tf

from .ops import mu_law_encode,optimizer_factory,SubPixelConvolution
from .mixture import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic
class WaveNetModel(object):
    def __init__(self,batch_size,dilations,filter_width,residual_channels,dilation_channels,skip_channels,quantization_channels=2**8,out_channels=30,
                 use_biases=False,scalar_input=False,global_condition_channels=None,
                 global_condition_cardinality=None,local_condition_channels=80,upsample_factor=None,legacy=True,residual_legacy=True,train_mode=True,drop_rate=0.0):

        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
        self.local_condition_channels=local_condition_channels
        self.upsample_factor=upsample_factor
        self.train_mode = train_mode
        self.out_channels = out_channels
        self.legacy=legacy
        self.residual_legacy=residual_legacy
        self.drop_rate = drop_rate
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        
        self.receptive_field = WaveNetModel.calculate_receptive_field(self.filter_width, self.dilations)
        
    @staticmethod
    def calculate_receptive_field(filter_width, dilations):
        # causal 때문에 length (T-1) + (여기서 계산되는 receptive_field만큼의  padding)  --> 최종 output의 길이가 T가 된다.
        receptive_field = (filter_width - 1) * sum(dilations) + 1  # 마지막 +1은 causal condition 때문에 1개 자른 것의 때문에 길이가 T-1인 되기 때문에 +1을 통해서 입력과 같은 길이 T가 된다.
        return receptive_field

    def _create_causal_layer(self, input_batch):
        with tf.name_scope('causal_layer'):
            
            if self.scalar_input:
                return tf.layers.conv1d(input_batch,filters=self.residual_channels,kernel_size=1,padding='valid',dilation_rate=1,use_bias=True)
            else:
                return tf.layers.conv1d(input_batch,filters=self.residual_channels,kernel_size=1,padding='valid',dilation_rate=1,use_bias=True)


    def _create_queue(self):
        # first layer(causal layer)나 local condition은 kernel_size = 1이므로, Queue가 필요없다.
        with tf.variable_scope('queue'):
            self.dilation_queue=[]
            for i,d in enumerate(self.dilations):
                q = tf.Variable(initial_value=tf.zeros(shape=[self.batch_size,d*(self.filter_width-1)+1,self.residual_channels], dtype=tf.float32), name='dilation_queue'.format(i), trainable=False)
                self.dilation_queue.append(q)
        
        # restore했을 때, Dilation_Queue,Causal_Queue는 0으로 initialization해야 한다.
        self.queue_initializer= tf.variables_initializer(self.dilation_queue)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,local_condition_batch,global_condition_batch):
        # input_batch는 train mode에서는 길이 줄어드는 것을 대비하여 padding이 되어 있다.
        with tf.variable_scope('dilation_layer'):
            residual =  input_batch
            if self.train_mode:
                # padding
                padding = (self.filter_width - 1)*dilation
                input_batch = tf.pad(input_batch, tf.constant([(0, 0), (padding, 0), (0, 0)]))

            else:
                self.dilation_queue[layer_index] =  tf.scatter_update(self.dilation_queue[layer_index],tf.range(self.batch_size),tf.concat([self.dilation_queue[layer_index][:,1:,:],input_batch],axis=1) )
                input_batch =  self.dilation_queue[layer_index]


            input_batch = tf.layers.dropout(input_batch,rate=self.drop_rate,training=self.train_mode)
            
            dilation_layer = tf.layers.Conv1D(filters=self.dilation_channels*2,kernel_size=self.filter_width,dilation_rate=dilation,padding='valid',use_bias=self.use_biases,name='conv_filter_gate')
            
            if self.train_mode:
                conv = dilation_layer(input_batch)
                conv_filter, conv_gate = tf.split(conv,2,axis=-1)
                
            else:
                
                dilation_layer.build((self.batch_size,1,input_batch.shape.as_list()[-1]))   # shape의 마지막만 중요함. kernel을 잡는데 마지막 차원만 사용됨
                
                linearized_weights = tf.reshape(dilation_layer.kernel,(-1,self.dilation_channels*2))
                input_batch = input_batch[:, 0::dilation, :]
                temp = tf.matmul(tf.reshape(input_batch,(self.batch_size,-1)), linearized_weights)
                if self.use_biases:
                    temp = tf.nn.bias_add(temp, dilation_layer.bias)                
                
                conv_filter, conv_gate = tf.split(tf.expand_dims(temp,1),2,axis=-1)
                
                            
            if global_condition_batch is not None:
                conv_filter += tf.layers.conv1d(global_condition_batch,filters=self.dilation_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="gc_filter")
                conv_gate += tf.layers.conv1d(global_condition_batch,filters=self.dilation_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="gc_gate")
    
            if local_condition_batch is not None:
                local_filter = tf.layers.conv1d(local_condition_batch,filters=self.dilation_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="lc_filter")
                local_gate = tf.layers.conv1d(local_condition_batch,filters=self.dilation_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="lc_gate")
                
                conv_filter += local_filter
                conv_gate += local_gate            
                
                    
            out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
    
            # The 1x1 conv to produce the residual output  == FC
            transformed = tf.layers.conv1d(out,filters=self.residual_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="dense")
    
            # The 1x1 conv to produce the skip output
            skip_contribution = tf.layers.conv1d(out,filters=self.skip_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="skip")
    

            # residual + transformed: 다음 단계의 입력으로 들어감
            if self.residual_legacy:
                out = (residual + transformed) * np.sqrt(0.5)
            else:
                out = residual + transformed
    
            return skip_contribution, out   # skip_contribution: 결과값으로 쌓임. 
    def create_upsample(self, local_condition_batch,upsample_type='SubPixel'):
        local_condition_batch = tf.expand_dims(local_condition_batch, [3])
        # local condition batch N H W C
        freq_axis_kernel_size = self.filter_width   # Rayhane-mamah 코드에서는 hyper parameter로 받음. frame(num_mels)에 적용되는 kernel_size임
        for i in range(len(self.upsample_factor)):
            if upsample_type =='SubPixel':
                
                # NN_init, NN_scaler <---- hyper parameter이지만, 여기서는 True, 0.3으로 고정
                # kernel_size: (3, hparams.freq_axis_kernel_size) 이렇게 되어 있는데, 왜 3인지 모르겠음. upsample_factor[i]로 대체. 
                # freq_axis_kernel_size는 hparams에 3으로 되어 있는데, 여기서는 filter_width로 처리  <---- frame(num_mels)에 적용되는 kernel_size임
                subpixel_layer = SubPixelConvolution(filters=1, kernel_size=(self.upsample_factor[i],freq_axis_kernel_size),padding='same', strides=(self.upsample_factor[i],1),
                                      NN_init=True, NN_scaler=0.3,up_layers=len(self.upsample_factor), name='SubPixelConvolution_layer_{}'.format(i))
                local_condition_batch = subpixel_layer(local_condition_batch)
            else:
                local_condition_batch = tf.layers.conv2d_transpose(local_condition_batch,filters=1, kernel_size=(self.upsample_factor[i], freq_axis_kernel_size),
                                                   strides=(self.upsample_factor[i],1),padding='same',use_bias=False,name='upsample_2D_{}'.format(i))
            
            local_condition_batch = tf.nn.relu(local_condition_batch)
            
            # for debugging
            #local_condition_batch = tf.Print(local_condition_batch,[tf.shape(local_condition_batch),"xx{}".format(i)])
        local_condition_batch = tf.squeeze(local_condition_batch, [3])
        
        return local_condition_batch
    def _create_network(self, input_batch,local_condition_batch, global_condition_batch):  
        '''Construct the WaveNet network.'''
        # global_condition_batch: (batch_size, 1, self.global_condition_channels)  <--- 가운데 1은 크기 1짜리 data FC대신에 conv1d를 적용하기 위해 강제로 넣었다고 봐야 한다.
        
        if self.train_mode==False:
            self._create_queue()
        
        
        current_layer = input_batch  # causal cut으로 길이 1이 줄어든 상태
           

        # Pre-process the input with a regular convolution
        current_layer = self._create_causal_layer(current_layer)  # 여전 모델에서는 길이가 줄었지만, 수정 후에는 길이 불변

        # Add all defined dilation layers.
        outputs = None
        with tf.variable_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations): # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                with tf.variable_scope('layer{}'.format(layer_index)):
                    
                    output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation,local_condition_batch,global_condition_batch)

                    if outputs is None:
                        outputs = output
                    else:
                        outputs = outputs + output
                        
                        if self.legacy:
                            outputs = outputs * np.sqrt(0.5)
                        
        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
             
            transformed1 = tf.nn.relu(outputs)
            conv1 = tf.layers.conv1d(transformed1,filters=self.skip_channels,kernel_size=1,padding="same",use_bias=self.use_biases)
    
            transformed2 = tf.nn.relu(conv1)
            if self.scalar_input:
                conv2 = tf.layers.conv1d(transformed2,filters=self.out_channels,kernel_size=1,padding="same",use_bias=self.use_biases)
            else:
                conv2 = tf.layers.conv1d(transformed2,filters=self.quantization_channels,kernel_size=1,padding="same",use_bias=self.use_biases)

        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(input_batch, depth=self.quantization_channels, dtype=tf.float32)  # (1, ?, 1) --> (1, ?, 1, 256)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)  # (1, ?, 1, 256) --> (1, ?, 256)
        return encoded

    def _embed_gc(self, global_condition):  # global_condition = global_condition_batch <---- data
        '''Returns embedding for global condition.
        :param global_condition: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding. The latter is
               experimental.
        :return: Embedding or None
        '''
        # global_condition: (N,)
        # self.global_condition_cardinality가 None이 아니며, global_condition 은 gc id이면 되고, 그렇지 않으면, global_condition은 embedding vector가 넘어와야 한다.
        embedding = None
        if self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = tf.get_variable('gc_embedding', [self.global_condition_cardinality, self.global_condition_channels], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(uniform=False))   # (2, 32)
            embedding = tf.nn.embedding_lookup(embedding_table,global_condition)
        elif global_condition is not None:
            # ... else the global_condition (if any) is already provided
            # as an embedding.

            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape())
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] == self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not match global_condition_channels {}.'.format(global_condition.get_shape(),
                                        self.global_condition_channels))
            embedding = global_condition

        if embedding is not None:
            embedding = tf.reshape(embedding,[self.batch_size, 1, self.global_condition_channels])

        return embedding


    def predict_proba_incremental(self, waveform,upsampled_local_condition=None, global_condition=None,name='wavenet'):
        """
        local_condition: upsampled local condition
        """


        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            
            if self.scalar_input:
                encoded = tf.reshape(waveform , [self.batch_size, -1, 1])  # (N,1,1)
            else:
                encoded = tf.one_hot(waveform, self.quantization_channels)
                encoded = tf.reshape(encoded, [self.batch_size,-1, self.quantization_channels])   # encoded shape=(N,1, 256)
            
            gc_embedding = self._embed_gc(global_condition)                   # --> shape=(1, 1, 32)
            
            
            # local condition
            if upsampled_local_condition is not None:
                upsampled_local_condition = tf.reshape(upsampled_local_condition , [self.batch_size, -1, self.local_condition_channels])
            
            raw_output = self._create_network(encoded,upsampled_local_condition,gc_embedding)        # 이것이 fast generation algorithm의 핵심  --> (batch_size, 1, 256)
            
            if self.scalar_input:
                out = tf.reshape(raw_output, [self.batch_size, -1, self.out_channels])
                proba = sample_from_discretized_mix_logistic(out)
            else:
                out = tf.reshape(raw_output, [self.batch_size, self.quantization_channels])
                proba = tf.cast(tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)

            return proba

    def add_loss(self, input_batch,local_condition=None, global_condition_batch=None, l2_regularization_strength=None,upsample_type=None, name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.variable_scope(name):
            # We mu-law encode and quantize the input audioform.
            # quantization_channels 크기의 one hot encoding을 적용한 예정. 16bit= 65536개였다면,  quantization_channels로 줄이는 효과가 있다.
            # mu law encoding은 bit를 단순히 줄이는 것보다 advanced된 방식으로 줄인다.
            # input_batch: (batch_size,?,1)  <-- 마지막 1은 channel 1을 의미
            encoded_input = mu_law_encode(input_batch, self.quantization_channels)  # "quantization_channels": 256   ---> (batch_size, ?, 1)

            gc_embedding = self._embed_gc(global_condition_batch) # (self.batch_size, 1, self.global_condition_channels) <--- 가운데 1은 강제로 reshape
            encoded = self._one_hot(encoded_input)      #  (1, ?, quantization_channels=256)
            if self.scalar_input:
                network_input = tf.reshape( tf.cast(input_batch, tf.float32), [self.batch_size, -1, 1])
            else:
                network_input = encoded
                
            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            if self.scalar_input:
                input = tf.slice(network_input, [0, 0, 0], [-1, network_input_width,1])
            else:
                input = tf.slice(network_input, [0, 0, 0], [-1, network_input_width, self.quantization_channels])


            # local condition
            if local_condition is not None:
                local_condition = self.create_upsample(local_condition,upsample_type)
                local_condition = tf.slice(local_condition, [0, 0, 0], [-1, network_input_width,self.local_condition_channels])

            raw_output = self._create_network(input,local_condition, gc_embedding)  # (batch_size, ?, quantization_channels=256) , (batch_size, 1, self.global_condition_channels)
            
            
            with tf.name_scope('loss'):
                # Cut off the samples corresponding to the receptive field
                # for the first predicted sample.
                
                # scalar input인 경우에도 target은 mu-law companding된 것이 된다.
                target_output = tf.slice(network_input , [0, 1, 0],[-1, -1, -1])   # [-1,-1,-1] --> 나머지 모두
                
                if self.scalar_input:
                    loss = discretized_mix_logistic_loss(raw_output, target_output,num_class=2**16, reduce=False)
                    reduced_loss = tf.reduce_mean(loss)                    
                else:
                    # 3 dim array의 loss를 계산학 위해, 2 dim으로 변환한다. batch와 time 부분을 합쳐서 2dim으로 변환
                    target_output = tf.reshape(target_output, [-1, self.quantization_channels])
                    prediction = tf.reshape(raw_output, [-1, self.quantization_channels])
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=target_output)
                    reduced_loss = tf.reduce_mean(loss)

                tf.summary.scalar('loss', reduced_loss)

                if l2_regularization_strength is None:
                    self.loss = reduced_loss
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)  for v in tf.trainable_variables() if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss + l2_regularization_strength * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    self.loss = total_loss

    def add_optimizer(self, hparams,global_step):
        '''Adds optimizer to the graph. Supposes that initialize function has already been called.
        '''
        with tf.variable_scope('optimizer'):
            hp = hparams

            learning_rate = tf.train.exponential_decay(hp.wavenet_learning_rate, global_step,hp.wavenet_decay_steps,hp.wavenet_decay_rate)

            #Adam optimization
            self.learning_rate = learning_rate
            optimizer = tf.train.AdamOptimizer(learning_rate)

            gradients, variables = zip(*optimizer.compute_gradients(self.loss))   # len(tf.trainable_variables()) = len(variables)
            self.gradients = gradients

            #Gradients clipping
            if hp.wavenet_clip_gradients:
                # Rayhane-mamah는 tf.clip_by_norm -> tf.clip_by_value 두 단계를 적용. 여기서는 tf.clip_by_global_norm
                
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)   # tf.clip_by_global_norm vs tf.clip_by_norm
            else:
                clipped_gradients = gradients

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                adam_optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),global_step=global_step)        
                
        #Add exponential moving average
        #https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        #Use adam optimization process as a dependency
        with tf.control_dependencies([adam_optimize]):
            #Create the shadow variables and add ops to maintain moving averages
            #Also updates moving averages after each update step
            #This is the optimize call instead of traditional adam_optimize one.
            assert tuple(tf.trainable_variables()) == variables #Verify all trainable variables are being averaged
            self.optimize = self.ema.apply(variables)                             
                
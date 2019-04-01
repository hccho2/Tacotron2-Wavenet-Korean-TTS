#  coding: utf-8
"""
- train data를 speaker를 분리된 디렉토리로 받아서, speaker id를 디렉토리별로 부과.
- file name에서 speaker id를 추론하는 방식이 아님.

"""

from __future__ import print_function

import argparse
import numpy as np
import os
import time
import traceback
from glob import glob
import tensorflow as tf
from tensorflow.python.client import timeline
from datetime import datetime
from wavenet import WaveNetModel,mu_law_decode
from datasets import DataFeederWavenet
from hparams import hparams
from utils import validate_directories,load,save,infolog,get_tensors_in_checkpoint_file,build_tensors_in_checkpoint_file,plot,audio

tf.logging.set_verbosity(tf.logging.ERROR)
EPSILON = 0.001
log = infolog.log

def eval_step(sess,logdir,step,waveform,upsampled_local_condition_data,speaker_id_data,mel_input_data,samples,speaker_id,upsampled_local_condition,next_sample,temperature=1.0):
    waveform = waveform[:,:1]
    
    sample_size = upsampled_local_condition_data.shape[1]
    last_sample_timestamp = datetime.now()
    start_time = time.time()
    for step2 in range(sample_size):  # 원하는 길이를 구하기 위해 loop sample_size
        window = waveform[:,-1:]  # 제일 끝에 있는 1개만 samples에 넣어 준다.  window: shape(N,1)
        

        prediction = sess.run(next_sample, feed_dict={samples: window,upsampled_local_condition: upsampled_local_condition_data[:,step2,:],speaker_id: speaker_id_data })


        if hparams.scalar_input:
            sample = prediction  # logistic distribution으로부터 sampling 되었기 때문에, randomness가 있다.
        else:
            # Scale prediction distribution using temperature.
            # 다음 과정은 config.temperature==1이면 각 원소를 합으로 나누어주는 것에 불과. 이미 softmax를 적용한 겂이므로, 합이 1이된다. 그래서 값의 변화가 없다.
            # config.temperature가 1이 아니며, 각 원소의 log취한 값을 나눈 후, 합이 1이 되도록 rescaling하는 것이 된다.
            np.seterr(divide='ignore')
            scaled_prediction = np.log(prediction) / temperature   # config.temperature인 경우는 값의 변화가 없다.
            scaled_prediction = (scaled_prediction - np.logaddexp.reduce(scaled_prediction,axis=-1,keepdims=True))  # np.log(np.sum(np.exp(scaled_prediction)))
            scaled_prediction = np.exp(scaled_prediction)
            np.seterr(divide='warn')
    
            # Prediction distribution at temperature=1.0 should be unchanged after
            # scaling.
            if temperature == 1.0:
                np.testing.assert_allclose( prediction, scaled_prediction, atol=1e-5, err_msg='Prediction scaling at temperature=1.0 is not working as intended.')
            
            # argmax로 선택하지 않기 때문에, 같은 입력이 들어가도 달라질 수 있다.
            sample = [[np.random.choice(np.arange(hparams.quantization_channels), p=p)] for p in scaled_prediction]  # choose one sample per batch
        
        waveform = np.concatenate([waveform,sample],axis=-1)   #window.shape: (N,1)

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            duration = time.time() - start_time
            print('Sample {:3<d}/{:3<d}, ({:.3f} sec/step)'.format(step2 + 1, sample_size, duration), end='\r')
            last_sample_timestamp = current_sample_timestamp
    
    print('\n')
    # Save the result as a wav file.    
    if hparams.input_type == 'raw':
        out = waveform[:,1:]
    elif hparams.input_type == 'mulaw':
        decode = mu_law_decode(samples, hparams.quantization_channels,quantization=False)
        out = sess.run(decode, feed_dict={samples: waveform[:,1:]})
    else:  # 'mulaw-quantize'
        decode = mu_law_decode(samples, hparams.quantization_channels,quantization=True)
        out = sess.run(decode, feed_dict={samples: waveform[:,1:]})          
        
        
    # save wav
    
    for i in range(1):
        wav_out_path= logdir + '/test-{}-{}.wav'.format(step,i)
        mel_path =  wav_out_path.replace(".wav", ".png")
        
        gen_mel_spectrogram = audio.melspectrogram(out[i], hparams).astype(np.float32).T
        audio.save_wav(out[i], wav_out_path, hparams.sample_rate)  # save_wav 내에서 out[i]의 값이 바뀐다.
        
        plot.plot_spectrogram(gen_mel_spectrogram, mel_path, title='generated mel spectrogram{}'.format(step),target_spectrogram=mel_input_data[i])  

def create_network(hp,batch_size,num_speakers,is_training):
    net = WaveNetModel(
        batch_size=batch_size,
        dilations=hp.dilations,
        filter_width=hp.filter_width,
        residual_channels=hp.residual_channels,
        dilation_channels=hp.dilation_channels,
        quantization_channels=hp.quantization_channels,
        out_channels =hp.out_channels,
        skip_channels=hp.skip_channels,
        use_biases=hp.use_biases,  #  True
        scalar_input=hp.scalar_input,
        global_condition_channels=hp.gc_channels,
        global_condition_cardinality=num_speakers,
        local_condition_channels=hp.num_mels,
        upsample_factor=hp.upsample_factor,
        legacy = hp.legacy,
        residual_legacy = hp.residual_legacy,
        drop_rate = hp.wavenet_dropout,
        train_mode=is_training)
    
    return net
def main():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    
    
    parser = argparse.ArgumentParser(description='WaveNet example network')
    
    DATA_DIRECTORY =  'D:\\hccho\\Tacotron-Wavenet-Vocoder-hccho\\data\\moon,D:\\hccho\\Tacotron-Wavenet-Vocoder-hccho\\data\\son'
    #DATA_DIRECTORY =  'D:\\hccho\\Tacotron-Wavenet-Vocoder-hccho\\data\\moon'
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY, help='The directory containing the VCTK corpus.')


    #LOGDIR = None
    LOGDIR = './/logdir-wavenet//train//2019-03-27T20-27-18'

    parser.add_argument('--logdir', type=str, default=LOGDIR,help='Directory in which to store the logging information for TensorBoard. If the model already exists, it will restore the state and will continue training. Cannot use with --logdir_root and --restore_from.')
    
    
    parser.add_argument('--logdir_root', type=str, default=None,help='Root directory to place the logging output and generated model. These are stored under the dated subdirectory of --logdir_root. Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,help='Directory in which to restore the model from. This creates the new model under the dated directory in --logdir_root. Cannot use with --logdir.')
    
    
    CHECKPOINT_EVERY = 1000   # checkpoint 저장 주기
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    
    
    parser.add_argument('--eval_every', type=int, default=2,help='Steps between eval on test data')
    
   
    
    config = parser.parse_args()  # command 창에서 입력받을 수 있는 조건
    config.data_dir = config.data_dir.split(",")
    
    try:
        directories = validate_directories(config,hparams)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from


    log_path = os.path.join(logdir, 'train.log')
    infolog.init(log_path, logdir)


    global_step = tf.Variable(0, name='global_step', trainable=False)

    if hparams.l2_regularization_strength == 0:
        hparams.l2_regularization_strength = None


    # Create coordinator.
    coord = tf.train.Coordinator()
    num_speakers = len(config.data_dir)
    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = hparams.silence_threshold if hparams.silence_threshold > EPSILON else None
        gc_enable = True  # Before: num_speakers > 1    After: 항상 True
        
        # AudioReader에서 wav 파일을 잘라 input값을 만든다. receptive_field길이만큼을 앞부분에 pad하거나 앞조각에서 가져온다. (receptive_field+ sample_size)크기로 자른다.
        reader = DataFeederWavenet(coord,config.data_dir,batch_size=hparams.wavenet_batch_size,gc_enable= gc_enable,test_mode=False)
        
        # test를 위한 DataFeederWavenet를 하나 만들자. 여기서는 딱 1개의 파일만 가져온다.
        reader_test = DataFeederWavenet(coord,config.data_dir,batch_size=1,gc_enable= gc_enable,test_mode=True,queue_size=1)
        
        

        audio_batch, lc_batch, gc_id_batch = reader.inputs_wav, reader.local_condition, reader.speaker_id


    # Create train network.
    net = create_network(hparams,hparams.wavenet_batch_size,num_speakers,is_training=True)
    net.add_loss(input_batch=audio_batch,local_condition=lc_batch, global_condition_batch=gc_id_batch, l2_regularization_strength=hparams.l2_regularization_strength,upsample_type=hparams.upsample_type)
    net.add_optimizer(hparams,global_step)



    run_metadata = tf.RunMetadata()

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))  # log_device_placement=False --> cpu/gpu 자동 배치.
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=hparams.max_checkpoints)  # 최대 checkpoint 저장 갯수 지정
    
    try:
        start_step = load(saver, sess, restore_from)  # checkpoint load
        if is_overwritten_training or start_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            zero_step_assign = tf.assign(global_step, 0)
            sess.run(zero_step_assign)
            start_step=0
    except:
        print("Something went wrong while restoring checkpoint. We will terminate training to avoid accidentally overwriting the previous model.")
        raise


    ###########

    reader.start_in_session(sess,start_step)
    reader_test.start_in_session(sess,start_step)
    
    ################### Create test network.  <---- Queue 생성 때문에, sess restore후 test network 생성
    net_test = create_network(hparams,1,num_speakers,is_training=False)
  
    if hparams.scalar_input:
        samples = tf.placeholder(tf.float32,shape=[net_test.batch_size,None])
        waveform = 2*np.random.rand(net_test.batch_size).reshape(net_test.batch_size,-1)-1
        
    else:
        samples = tf.placeholder(tf.int32,shape=[net_test.batch_size,None])  # samples: mu_law_encode로 변환된 것. one-hot으로 변환되기 전. (batch_size, 길이)
        waveform = np.random.randint(hparams.quantization_channels,size=net_test.batch_size).reshape(net_test.batch_size,-1)
    upsampled_local_condition = tf.placeholder(tf.float32,shape=[net_test.batch_size,hparams.num_mels])  
    
        

    speaker_id = tf.placeholder(tf.int32,shape=[net_test.batch_size])  
    next_sample = net_test.predict_proba_incremental(samples,upsampled_local_condition,speaker_id)  # Fast Wavenet Generation Algorithm-1611.09482 algorithm 적용

        
    sess.run(net_test.queue_initializer)
    



    # test를 위한 placeholder는 모두 3개: samples,speaker_id,upsampled_local_condition
    # test용 mel-spectrogram을 하나 뽑자. 그것을 고정하지 않으면, thread가 계속 돌아가면서 data를 읽어온다.  reader_test의 역할은 여기서 끝난다.

    mel_input_test, speaker_id_test = sess.run([reader_test.local_condition,reader_test.speaker_id])


    with tf.variable_scope('wavenet',reuse=tf.AUTO_REUSE):
        upsampled_local_condition_data = net_test.create_upsample(mel_input_test,upsample_type=hparams.upsample_type)
        upsampled_local_condition_data_ = sess.run(upsampled_local_condition_data)  # upsampled_local_condition_data_ 을 feed_dict로 placehoder인 upsampled_local_condition에 넣어준다.

    ######################################################
    
    
    start_step = sess.run(global_step)
    step = last_saved_step = start_step
    try:        
        
        while not coord.should_stop():
            
            start_time = time.time()
            if hparams.store_metadata and step % 50 == 0:
                # Slow run that stores extra information for debugging.
                log('Storing metadata')
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                step, loss_value, _ = sess.run([global_step, net.loss, net.optimize],options=run_options,run_metadata=run_metadata)

                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))
            else:
                step, loss_value, _ = sess.run([global_step,net.loss, net.optimize])

            duration = time.time() - start_time
            log('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
            
            
            if step % config.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step
                
                
            if step % config.eval_every == 0:  # config.eval_every
                eval_step(sess,logdir,step,waveform,upsampled_local_condition_data_,speaker_id_test,mel_input_test,samples,speaker_id,upsampled_local_condition,next_sample)
            
            if step >= hparams.num_steps:
                # error message가 나오지만, 여기서 멈춘 것은 맞다.
                raise Exception('End xxx~~~yyy')
            
    except Exception as e:
        print('finally')
        log('Exiting due to exception: %s' % e, slack=True)
        #if step > last_saved_step:
        #    save(saver, sess, logdir, step)        
        traceback.print_exc()
        coord.request_stop(e)


if __name__ == '__main__':
    main()
    traceback.print_exc()
    print('Done')

# -*- coding: utf-8 -*-
import re,json,sys,os
import tensorflow as tf
from tqdm import tqdm
from contextlib import closing
from multiprocessing import Pool
from collections import namedtuple
from datetime import datetime, timedelta
from shutil import copyfile as copy_file
from tensorflow.python import pywrap_tensorflow

PARAMS_NAME = "params.json"
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
LOGDIR_ROOT_Wavenet = './logdir-wavenet'


class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []
def prepare_dirs(config, hparams):
    if hasattr(config, "data_paths"):
        config.datasets = [os.path.basename(data_path) for data_path in config.data_paths]
        dataset_desc = "+".join(config.datasets)

    if config.load_path:
        config.model_dir = config.load_path
    else:
        config.model_name = "{}_{}".format(dataset_desc, get_time())
        config.model_dir = os.path.join(config.log_dir, config.model_name)

        for path in [config.log_dir, config.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    if config.load_path:
        load_hparams(hparams, config.model_dir)
    else:
        setattr(hparams, "num_speakers", len(config.datasets))

        save_hparams(config.model_dir, hparams)
        copy_file("hparams.py", os.path.join(config.model_dir, "hparams.py"))

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    #ckpt = get_most_recent_checkpoint(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    if not os.path.exists(logdir):
        os.makedirs(logdir)    
    return logdir


def validate_directories(args,hparams):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT_Wavenet
        
        
    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))
        save_hparams(logdir, hparams)
        copy_file("hparams.py", os.path.join(logdir, "hparams.py"))
    else:
        load_hparams(hparams, logdir)
        
    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }
def save_hparams(model_dir, hparams):
    param_path = os.path.join(model_dir, PARAMS_NAME)

    info = eval(hparams.to_json(),{'false': False, 'true': True, 'null': None})
    write_json(param_path, info)

    print(" [*] MODEL dir: {}".format(model_dir))
    print(" [*] PARAM path: {}".format(param_path))
    
def write_json(path, data):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)

def load_hparams(hparams, load_path, skip_list=[]):
    # log dir에 있는 hypermarameter 정보를 이용해서, hparams.py의 정보를 update한다.
    path = os.path.join(load_path, PARAMS_NAME)

    new_hparams = load_json(path)
    hparams_keys = vars(hparams).keys()

    for key, value in new_hparams.items():
        if key in skip_list or key not in hparams_keys:
            print("Skip {} because it not exists".format(key))  #json에 있지만, hparams에 없다는 의미
            continue

        if key not in ['xxxxx',]:  # update 하지 말아야 할 것을 지정할 수 있다.
            original_value = getattr(hparams, key)
            if original_value != value:
                print("UPDATE {}: {} -> {}".format(key, getattr(hparams, key), value))
                setattr(hparams, key, value)
def load_json(path, as_class=False, encoding='euc-kr'):
    with open(path,encoding=encoding) as f:
        content = f.read()
        content = re.sub(",\s*}", "}", content)
        content = re.sub(",\s*]", "]", content)

        if as_class:
            data = json.loads(content, object_hook=\
                    lambda data: namedtuple('Data', data.keys())(*data.values()))
        else:
            data = json.loads(content)

    return data    
def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))

    #latest_checkpoint=checkpoint_paths[0]
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint

def add_prefix(path, prefix):
    dir_path, filename = os.path.dirname(path), os.path.basename(path)
    return "{}/{}.{}".format(dir_path, prefix, filename)

def add_postfix(path, postfix):
    path_without_ext, ext = path.rsplit('.', 1)
    return "{}.{}.{}".format(path_without_ext, postfix, ext)

def remove_postfix(path):
    items = path.rsplit('.', 2)
    return items[0] + "." + items[2]

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def parallel_run(fn, items, desc="", parallel=True):
    results = []

    if parallel:
        with closing(Pool(10)) as pool:
            for out in tqdm(pool.imap_unordered(fn, items), total=len(items), desc=desc):
                if out is not None:
                    results.append(out)
    else:
        for item in tqdm(items, total=len(items), desc=desc):
            out = fn(item)
            if out is not None:
                results.append(out)

    return results
def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)
        
def str2bool(v):
    return v.lower() in ('true', '1')

def remove_file(path):
    if os.path.exists(path):
        print(" [*] Removed: {}".format(path))
        os.remove(path)

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")
def get_git_diff():
    return subprocess.check_output(['git', 'diff']).decode("utf-8")

def warning(msg):
    print("="*40)
    print(" [!] {}".format(msg))
    print("="*40)
    print()		

def get_tensors_in_checkpoint_file(file_name,all_tensors=True,tensor_name=None):
    # checkpoint 파일로 부터 복구
    # e.g  file_name: 'D:\\hccho\\Tacotron-2-hccho\\model.ckpt-155000'
    varlist=[]
    var_value =[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    trainable_variables_names = [v.name[:-2] for v in tf.trainable_variables()]  # 끝부분의 ':0' 제외
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            if key in trainable_variables_names:   # hccho
                varlist.append(key)
                var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)

def build_tensors_in_checkpoint_file(loaded_tensors):
    # 현재 tensor graph에 있는 tensor중에서 loaded_tensors에 있는 tensor name을 가져온다.
    full_var_list = list()
    # Loop all loaded tensors
    for i, tensor_name in enumerate(loaded_tensors[0]):
        # Extract tensor
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
        except:
            print('Not found: '+tensor_name)
        full_var_list.append(tensor_aux)
    return full_var_list
"""
    # restore egample 모델을 변형했을 때, 기존 ckpt로부터 중복되는 trainable_varaibles 복구.
    CHECKPOINT_NAME = 'D:\\hccho\\Tacotron-2-hccho\\ver1\\logdir-wavenet\\train\\2019-03-22T23-08-16\\model.ckpt-155000'
    restored_vars  = get_tensors_in_checkpoint_file(file_name=CHECKPOINT_NAME)
    tensors_to_load = build_tensors_in_checkpoint_file(restored_vars)
    loader = tf.train.Saver(tensors_to_load)
    loader.restore(sess, CHECKPOINT_NAME)
    
    new_saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=hparams.max_checkpoints)  # 최대 checkpoint 저장 갯수 지정
    save(new_saver, sess, logdir, 0)
    exit()

"""



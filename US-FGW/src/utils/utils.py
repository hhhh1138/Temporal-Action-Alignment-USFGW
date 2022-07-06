from ..home import get_project_base
from .dataset import Dataset 
import os
import numpy as np
from glob import glob
import json
import sys

BASE = get_project_base()

def get_dataset_paths(data, split=1): 
    if data == "breakfast":
        map_fname = BASE + '../data/Breakfast/mapping.txt'
        dataset_dir = BASE + '../data/Breakfast/'
        train_split_fname = BASE + '../data/Breakfast/split%d.train' % split
        test_split_fname = BASE + '../data/Breakfast/split%d.test' % split
    elif data == "crosstask":
        # assert split == 1
        map_fname = BASE + '../data/CrossTask/mapping.txt'
        dataset_dir = BASE + '../data/CrossTask/'
        train_split_fname = BASE + '../data/CrossTask/split%d.train' % split
        test_split_fname = BASE + '../data/CrossTask/split%d.test' % split
    elif data == "hollywood":
        map_fname = BASE + '../data/Hollywood/mapping.txt'
        dataset_dir = BASE + '../data/Hollywood/'
        train_split_fname = BASE + '../data/Hollywood/split%d.train' % split
        test_split_fname = BASE + '../data/Hollywood/split%d.test' % split
    
    return map_fname, dataset_dir, train_split_fname, test_split_fname


def create_dataset(args):
    map_fname, dataset_dir, train_split_fname, test_split_fname = get_dataset_paths(args.data, args.split)
    print("load_data_from", dataset_dir)

    ### read label2index mapping and index2label mapping ###########################
    label2index, index2label = load_action_mapping(map_fname)

    ### read training data #########################################################
    print('reading data...')
    with open(train_split_fname, 'r') as f:
        video_list = f.read().split('\n')[0:-1]

    dataset = Dataset(dataset_dir, video_list, label2index, shuffle=True)
    print("Number of training data", len(video_list))
    print(dataset)

    with open(test_split_fname, 'r') as f:
        test_video_list = f.read().split('\n')[0:-1]
    test_dataset = Dataset(dataset_dir, test_video_list, label2index, shuffle=False)
    return dataset_dir, label2index, index2label, dataset, test_dataset


def load_action_mapping(map_fname):
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    return label2index, index2label


def generate_exp_name(args):
    if not isinstance(args, dict):
        args = vars(args)

    exp = []

    # exp.append( '%s' % str(args['model_type']) )
    # exp.append( '%s' % str(args['task_type']) )
    # if args.get('lr', 0.01) != 0.01:
    #     exp.append( 'LR%s' % str(args['lr']) )
    # exp.append( 'SampleRate%s' % str(args['sample_rate']) )
    # exp.append( 'Pred-%s' % str(args['predict_type']) )
    # exp.append( 'loops%s' % str(args['badmm_loops']) )
    # exp.append( 'error-bound%s' % str(args['badmm_error_bound']) )
    # exp.append( 'Lambdav%s' % str(args['Lambdav']) )
    # exp.append( 'Lambdaw%s' % str(args['Lambdaw']) )
    # exp.append( 'Gamma%s' % str(args['gamma']) )
    # exp.append( 'Beta%s' % str(args['beta']) )
    # exp.append( 'B%s' % str(args['b']) )
    # exp.append( 'Tau%s' % str(args['tau']) )

    exp = "_".join(exp)
    return exp


class Recorder():

    def __init__(self):
        self.dict = {}

    def get(self, key):
        if key not in self.dict:
            self.dict[key] = list()
        return self.dict[key]

    def append(self, key, value):
        self.get(key).append(value)

    def extend(self, key, value):
        self.get(key).extend(value)

    def reset(self, key):
        self.dict[key] = []

    def mean_reset(self, key):
        mean = np.mean(self.get(key))
        self.reset(key)
        return mean

    def get_reset(self, key):
        val = self.get(key)
        self.reset(key)
        return val

def get_load_iteration(resume=None, savedir=None):
    
    if resume == "max":
        network_ckpts = glob(savedir + "/network.iter-*.net")
        iterations = [ int(os.path.basename(f)[:-4].split("-")[-1]) for f in network_ckpts ]
        if len(iterations) > 0: 
            load_iteration = max(iterations)
        else:
            load_iteration = 0 # no checkpoint to use
    else:
        load_iteration = os.path.basename(resume)
        load_iteration = int(load_iteration.split('.')[1].split('-')[1])
        savedir = os.path.dirname(resume)

    net = os.path.join(savedir, 'network.iter-' + str(load_iteration) + '.net')
    prior = os.path.join(savedir, 'prior.iter-' + str(load_iteration) + '.txt')
    length = os.path.join(savedir, 'lengths.iter-' + str(load_iteration) + '.txt')
    buffer = os.path.join(savedir, 'buffer.iter-%d.pk' % load_iteration)

    return load_iteration, net, prior, length, buffer 

def create_logdir(log_dir, check_exist=True):
    if check_exist and os.path.exists(log_dir):
        print('\nWARNING: log_dir exists %s\n' % log_dir)

    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    return log_dir, ckpt_dir


def create_rerun_script(fname):
    with open(fname, 'w') as fp:
        cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        fp.write( "cd " + os.getcwd() + '\n' )
        fp.write("PY="+ sys.executable +'\n')

        if cuda_device:
            cuda_prefix = "CUDA_VISIBLE_DEVICES=%s " % cuda_device
        else:
            cuda_prefix = ""

        fp.write("%s$PY %s\n"%(cuda_prefix, " ".join(sys.argv)))


def log_param(info, args):
    info('============')

    if not isinstance(args, dict):
        args = vars(args)

    keys = sorted(args.keys())
    for k in keys:
        info( "%s: %s" % (k, args[k]) )

    info('============')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def prepare_save_env(logdir, exp_name, args=None, check_exist=True):

    logParentDir = os.path.join(logdir, exp_name)
    logDir, ckptDir = create_logdir(logParentDir, check_exist)

    rerun_fname = os.path.join(logDir, "run.sh")
    create_rerun_script(rerun_fname)


    if args:
        log_param(print, args)
        argSaveFile = os.path.join(logDir, 'args.json')
        with open(argSaveFile, 'w') as f:
            if not isinstance(args, dict):
                args = vars(args)
            json.dump(args, f, indent=True)

    return logDir, ckptDir


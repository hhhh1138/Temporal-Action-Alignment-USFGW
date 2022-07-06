#!/usr/bin/python3

from .utils.utils import generate_exp_name, prepare_save_env, create_dataset
from .utils.model import AE_video_MLP, AE_word_breakfast_MLP, AE_word_hollywood_MLP, AE_word_crosstask_MLP
from .utils.ae import train_model
from .home import get_project_base
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data', default=None, type=str)   # 'breakfast' / 'crosstask' / 'hollywood'
parser.add_argument('--split', default=1, type=int)     # 1 - 4
parser.add_argument('--M', default=3000, type=int, help='number of total iterations to run')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--gpu', default='3', type=str)
parser.add_argument('--exp', default='tmp', type=str, help='postfix of experiment name')
parser.add_argument('--xv-dim', type=int, default=64, help='video input dimension')
parser.add_argument('--xw-dim', type=int, default=64, help='word input dimension')
parser.add_argument('--z-dim', type=int, default=8, help='latent dimension') # ..
parser.add_argument('--K', type=int, default=10, help='the number of Gaussian components')
parser.add_argument('--model-type', type=str, default='deterministic', help='the type of autoencoder')
parser.add_argument('--sample_rate', type=int, default=10, help='sample rate for the video frames')
parser.add_argument('--gamma', type=float, default=1.0,help='the weight of regularizer')
parser.add_argument('--b', type=float, default=1.0,help='the weight used for computing kernel matrix')
parser.add_argument('--Lambdav', type=float, default=0.1,help='the weight of distance between the normalized indices of frames/texts,')
parser.add_argument('--Lambdaw', type=float, default=0 ,help='the weight of distance between the normalized indices of frames/texts,')
parser.add_argument('--task-type', type=str, default='transcript', help='the type of task')# tasks: 'transcript' and 'set'
parser.add_argument('--loss-type', type=str, default='CrossEntropy', help='the type of loss') # MSE, MAE, BCE, CrossEntropy
parser.add_argument('--autoencoder-type', type=str, default='MLP', help='the type of autoencoder') # 'MLP'
parser.add_argument('--beta', type=float, default=0.1, help='the trade-off in fgw between w and gw')
parser.add_argument('--tau', type=float, default=0.1, help='the weight of kl divergence in fgw')
parser.add_argument('--badmm-loops', type=int, default=3000, help='the iteration number in badmm')
parser.add_argument('--badmm-error-bound', type=float, default=1e-3, help='the iteration error bound in badmm')
parser.add_argument('--badmm-rho', type=float, default=1.0, help='a hyperpara controlling rate of convergence in badmm')
parser.add_argument('--predict-type', type=str, default='OT', help='the method used in predict section') # 'OT' / 'straight'

parser.add_argument('--test-every', default=500, type=int, help='run testdataset and compute metrics')
parser.add_argument('--print-every', default=1, type=int, help='the frequency of displaying the train loss')
parser.add_argument('--enable-grammar', default=0, type=int, help='whether uses the grammar tool 0/1')
parser.add_argument('--enable-spectral', default=1, type=int, help='whether uses the spectral distance 0/1')
parser.add_argument('--enable-contrastive-learning', default=1, type=int, help='whether uses the spectral distance 0/1')

if __name__ == '__main__':
    
    args = parser.parse_args()

    BASE = get_project_base()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

    # generate save path postfix, and save it in args.exp
    prefix = "Split%d/" % args.split
    expgroup = generate_exp_name(vars(args))

    if len(args.exp) > 0:
        args.exp = prefix + expgroup + "_" + args.exp
    else:
        args.exp = prefix + expgroup
    
    base_log_dir = os.path.join(BASE, "log", args.data)
    # logdir => savedir + test_result 
    # savedir => save network params
    logdir, savedir = prepare_save_env(base_log_dir, args.exp, args)

    dataset_dir, label2index, index2label, dataset, test_dataset = create_dataset(args)
    print('Dataset created successfully!')

    args.xw_dim = len(label2index) # word input dimension

    if args.autoencoder_type == 'MLP':
        model_v = AE_video_MLP(z_dim=args.z_dim, model_type=args.model_type, sample_rate=args.sample_rate, device=device)
        if args.data == 'breakfast':
            model_w = AE_word_breakfast_MLP(z_dim=args.z_dim, model_type=args.model_type)
        elif args.data == 'hollywood':
            model_w = AE_word_hollywood_MLP(z_dim=args.z_dim, model_type=args.model_type)
        elif args.data == 'crosstask':
            model_w = AE_word_crosstask_MLP(z_dim=args.z_dim, model_type=args.model_type)
        else:
            print('args data input error!!!')
    else:
        print('args autoencoder_type input error!!!')

    train_model(model_v, model_w, dataset, test_dataset, device, args, label2index, index2label, logdir, savedir)
    
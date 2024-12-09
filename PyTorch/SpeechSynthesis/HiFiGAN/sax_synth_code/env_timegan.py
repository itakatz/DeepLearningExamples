from ssynth.dataset import EnvelopesDataset, get_env_train_val_data, gCFG
from ssynth.utils import midi

import os
import glob
import numpy as np
import pandas as pd
import librosa
import pickle
#from scipy.interpolate import UnivariateSpline
#from scipy.signal import butter, sosfiltfilt

import math
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

#--- I created symlink to the TimeGAN_pytorch_fork folder, so no need to add ~/git_repos to path
#import sys
#sys.path.append('/home/mlspeech/itamark/git_repos')



'''
class PositionalEncoding():
	def __init__(self, d_model, max_len: int = 5000):
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        self.pe = np.zeros((max_len, 1, d_model), dtype = np.float32)
        self.pe[:, 0, 0::2] = np.sin(position * div_term)
        self.pe[:, 0, 1::2] = np.cos(position * div_term)
	
	def encode(self, x):
        n = x.shape[0]
        return x + self.pe[:n]
'''

def save_features_to_cache(cache_dir, cfg, n_workers = 32):
    ''' just read all samples from the dataset class, so features are saved to cache
    '''
    from multiprocessing.pool import Pool
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader, val_loader = get_env_train_val_data(cache_dir, cfg.batch_size, cfg.sample_sec, cfg.history_len)
    pickle.dump(cfg, open(f'{cache_dir}/config.pickle', 'wb'))

    for ds in [train_loader.dataset, val_loader.dataset]:
        N = len(ds)
        with Pool(n_workers) as pool:
            pool.map(ds.__getitem__ , range(N))

if __name__ == '__main__':
    import sys
    from TimeGAN_pytorch_fork.options import Options
    from TimeGAN_pytorch_fork.lib.env_timegan import EnvelopeTimeGAN

    #--- mimic input args
    # input_args = f'env_timegan.py --num_layer 5 --hidden_dim 64 --latent_dim 16 --embedding_dim 32 --batch_size {gCFG.batch_size} --outf results/2023_14_12_test --model EnvelopeTimeGAN --name test3'
    if len(sys.argv) == 1:
        #--- set input args only for running from ipython
        input_args = f'env_timegan.py --calc_z_grad --num_layer 6 --num_layer_gen 6 --num_layer_discrim 3 --hidden_dim 32 --latent_dim 64 --embedding_dim 32 --batch_size {gCFG.batch_size} --outf results/2024_03_12 --model EnvelopeTimeGAN --name lyr_4g4d3_ldim_64'
        sys.argv = input_args.split()
    opt = Options().parse()
    
    #--- data loaders
    cache_dir = 'feature_cache2'
    workers = 0 #opt.workers # using more then 1 process for loading just makes it slower (overhead of data to/from sub-process?)
    train_loader, val_loader = get_env_train_val_data(cache_dir, gCFG.batch_size, gCFG.sample_sec, gCFG.history_len, workers)
    
    #--- different no. of epoch for embed/supervised and for joint training
    opt.num_epochs_es = 150 #250 #opt.iteration
    opt.num_epochs = 2000 # opt.iteration
    #opt.batch_size = gCFG.batch_size

    x, xout, t, note_id, note_en, is_note, _ = train_loader.dataset[0] #--- get a sample for the dims
    opt.seq_len = x.shape[0] #167 # 86 - history_len + 1 #344*2+1
    opt.z_dim = x.shape[1] #history_len #1 # number of features per sequence frame
    opt.z_dim_out = xout.shape[1] #1

    #opt.batch_size = batch_size
    #opt.module = 'gru'
    #opt.outf = './output_TMP'
    opt.average_seq_before_loss = True
    opt.generator_loss_moments_axis = 1 # use "0" to calculate along batch (original impl, after bug fix), or "1" to calculate along the sequence (makes more sense)
    #--- load autoencode and supervisor (aka AES) from disk to start from joint training
    AES_checkpoint = 'results/2024_03_07/lyr_3_ldim_8/train/weights'
    AES_epoch = 200
    joint_train_only = False #True

    #--- to load model:
    #opt.resume = 'results/2023_17_12_test/test4_mean_bce/train/weights'
    #opt.resume_epoch = 0

    # opt.resume='results/2023_31_12_ldim2/bug_fix_and_smooth_loss1/train/weights'
    # opt.resume_epoch=499
    
    #opt.resume = 'results/2024_04_07/lyr_3_ldim_8/train/weights'
    #opt.resume_epoch = 1300

    model = EnvelopeTimeGAN(opt, train_loader, val_loader)
    if joint_train_only:
        model.load_AE_and_S(AES_checkpoint, AES_epoch)
    model.max_seq_len = opt.seq_len
    print(f'Options:\n{opt}')
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model params:')
    tot = 0
    for net in [model.net_note_embed, model.nete, model.netr, model.nets, model.netg, model.netd]:
        num_params = count_parameters(net)
        tot += num_params
        print(f'{type(net).__name__}: number of params: {num_params}')
    print(f'total params: {tot}')
            
    model.train(joint_train_only = joint_train_only)

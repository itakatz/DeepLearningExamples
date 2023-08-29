#!/usr/bin/env python
# coding: utf-8

import sys
import time
import pickle
import torch
import numpy as np
import glob
import librosa
import pandas as pd

import os
from mido import MidiFile
import mido
from scipy.interpolate import interp1d

#--- import HiFiGAN modules
#sys.path.append('../')

#==============================================
#====== sax_synth imports =====================

from ssynth import set_python_path
from ssynth.utils.melspec import MelSpec
from ssynth.utils.synthesis import additive_synth_sawtooth
from ssynth.utils.midi import MidiUtils

#==============================================

import torch.nn.functional as F
from tqdm import tqdm

import models
#import common.layers as layers 
from common.utils import load_wav #--- use same method that is used in hifigan for loading audio
#from hifigan.data_function import mel_spectrogram
from hifigan.models import Denoiser

#get_ipython().run_line_magic('matplotlib', 'notebook')
#import matplotlib.pyplot as plt
import librosa.display

#import IPython.display as ipd
#ipd.display(ipd.HTML("<style>.container { width:85% !important; }</style>"))

def load_generator_model(model_path, device = 'cuda'):
    DEVICE = device #'cuda' # 'cpu' or 'cuda'       
    #assert DEVICE == 'cuda', 'ERROR: cpu not supported yet (mel code assumes torch tensors)'
    
    #m_path = '../results/2023_01_20_hifigan_ssynth44khz_synthesized_input/hifigan_gen_checkpoint_10000.pt'
    #m_path = '../results/2023_05_15_hifigan_ssynth44khz_synthesized_input_16k_spl0.5/hifigan_gen_checkpoint_3000.pt'
    #m_path = '../results/2023_05_28_hifigan_ssynth44khz_synthesized_input_16k_spl0.5_nonorm/hifigan_gen_checkpoint_3000.pt'
    m_path = model_path
    map_loc = None if DEVICE == 'cuda' else torch.device('cpu')
    
    checkpoint = torch.load(m_path, map_location = map_loc)
    train_config = checkpoint['train_setup']
    sampling_rate = train_config['sampling_rate']
    gen_config = checkpoint['config']
    gen_config['num_mel_filters'] = train_config['num_mels']
    
    gen = models.get_model('HiFi-GAN', gen_config, DEVICE, forward_is_infer = True)
    gen.load_state_dict(checkpoint['generator'])
    gen.remove_weight_norm()
    gen.eval()
    
    denoiser = Denoiser(gen, win_length = train_config['win_length'], num_mel_filters = train_config['num_mels']).to(DEVICE)

    return gen, denoiser, train_config
    
def array_to_torch(x):
    x = torch.FloatTensor(x.astype(np.float32))
    x = torch.autograd.Variable(x, requires_grad = False)
    x = x.unsqueeze(0)
    return x   

#--- simple wrapper to apply HiFiGAN generator to input audio, using a given MEL spec method
def generate_from_audio(x, hifigan_gen, return_numpy_arr = True):
    ''' audio -> mel spectrum -> HifiGAN generator
    '''
    x = array_to_torch(x)    
    mel = g_mel.get_spec(x)

    if torch.cuda.is_available():
        mel = mel.cuda()
    x_hat = hifigan_gen(mel)
    if return_numpy_arr:
        x_hat = x_hat[0].cpu().detach().numpy()[0]
    
    return x_hat

def run_on_validation_set(gen, denoiser, flist_path, num_files = None, return_file_index = None, verbose = False):
    ''' load wav from validation set, get mel and apply model
        Note: synthesis method should fit the one used to train the model (i.e., "10 harmonics" or "16 khz" etc.)
    '''
    yret = {'y': None, 'y_hat': None, 'y_hat_den': None}
    
    flist_validation = open(f'{flist_path}/ssynth_audio_val.txt', 'r').readlines()
    flist_validation = [fnm.rstrip() for fnm in flist_validation]
    
    flist_train = open(f'{flist_path}/ssynth_audio_train.txt', 'r').readlines()
    flist_train = [fnm.rstrip() for fnm in flist_train]
    
    flist = flist_validation
    if num_files is not None:
        flist = flist[:num_files] #flist_train #
        
    n_files = len(flist)

    #wav_fnm = '../data_ssynth/wavs_synth_10h/01_Free_Improv_dynamic_mic_phrase000.wav'
    synth_wavs_folder = 'wavs_synth_16k_spl0.5' # 'wavs_synth_10h'
    times_lens = []
    mel_loss = np.zeros(n_files)
    mel_len = np.zeros(n_files)
    for file_index in tqdm(range(n_files)): #[5] #1
        wav_fnm_target = flist[file_index]
        y_target, sr, sample_type = load_wav(f'../data_ssynth/{wav_fnm_target}')
    
        wav_fnm = flist[file_index].replace('wavs/', f'{synth_wavs_folder}/')
        y, sr, sample_type = load_wav(f'../data_ssynth/{wav_fnm}')
    
        if sample_type == 'PCM_24':
            max_wav_value = 2**31 # data type in this case is int32
        elif sample_type == 'PCM_16':
            max_wav_value = 2**15
    
        #--- convert to float in [-1., 1.]
        y = y.astype(np.float32) / np.float32(max_wav_value)
        y_target = y_target.astype(np.float32) / np.float32(max_wav_value)

        t0 = time.time()
        y_hat = generate_from_audio(y, gen, return_numpy_arr = False)
        t1 = time.time()
        times_lens.append((t1 - t0, len(y) / sr))
        y_hat_den = denoiser(y_hat.squeeze(1), denoising_strength)
        y_hat = y_hat[0].cpu().detach().numpy()[0]
        y_hat_den = y_hat_den[0].cpu().detach().numpy()[0]

        if return_file_index ==  file_index:
            #--- choose file index to return
            yret['y'], yret['y_hat'], yret['y_hat_den'] = y_target, y_hat, y_hat_den
    
        mel_target = g_mel.get_spec(array_to_torch(y_target)).squeeze(0)
        mel_hat = g_mel.get_spec(array_to_torch(y_hat)).squeeze(0)
        mloss = F.l1_loss(mel_target, mel_hat)
        mel_loss[file_index] = mloss
        mel_len[file_index] = mel_target.shape[1]
        if verbose:
            print(f'file {file_index}/{n_files}: mel loss {mloss}')
    
    #--- compare mel spec of generated and GT
    # fig, ax = plt.subplots(1,2,figsize = (12,4), sharex=True, sharey=True)
    # ax[0].imshow(mel_target, aspect='auto',interpolation='none',origin='lower')
    # ax[1].imshow(mel_hat, aspect='auto',interpolation='none',origin='lower')
    # fig, ax = plt.subplots(figsize = (12,4))
    # k1,k2 = 360,450 #317 #90
    # ax.plot(mel_target[:,k1:k2].mean(1))
    # ax.plot(mel_hat[:,k1:k2].mean(1))
    # ax.grid()
    
    return mel_loss, mel_len, yret, times_lens

def synthetic2octaves(gen, denoiser, sampling_rate):
    ''' I define a naive ADSR envelopes with straight lines, probably not the best option
        TODO move to utils/synthesis.py - ??
    '''
    #range_notes = ['Db3', 'A5'] #['C3', 'A#5'] # alto sax range is ['Db3', 'A5'], take half-step below/above
    #alto_sax_range = librosa.note_to_hz(range_notes)
    
    #--- envelope parameters
    note_len_samples = 24000 #20000 #20000
    onset_samples = 4500 #3000
    amp = 0.03
    amp_sustain = 0.8 # decay envelope to this relative level at the end of the note
    freq_glide_level = 0.7 #--- during onset, glide into target frequency starting at this pitch (relative)
    alto_sax_min_hz = MidiUtils.alto_sax_range_hz[0]
    alto_sax_max_hz = MidiUtils.alto_sax_range_hz[1]
    
    freq = np.zeros(note_len_samples)
    env = np.zeros(note_len_samples)
    
    #--- single note envelope
    env_single = np.r_[np.linspace(0, 1, onset_samples),  np.linspace(1, amp_sustain, note_len_samples - onset_samples)]
    env_single = env_single ** 3
    env_single *= amp

    #--- major scale in the alto sax range
    for note in ['D3', 'E3', 'F#3', 'G3', 'A3', 'B3', 'C#4', 'D4', 'E4', 'F#4', 'G4', 'A4', 'B4', 'C#5', 'D5', 'E5', 'F#5', 'G5', 'A5']:
        f0 = librosa.note_to_hz(note)
        freq_env = np.ones(note_len_samples)
        freq_env[:onset_samples] *= np.linspace(freq_glide_level, 1, onset_samples)
        
        freq = np.r_[freq, f0 * freq_env]
        env = np.r_[env, env_single]
        
    freq = np.r_[freq, np.zeros(note_len_samples)]
    freq[freq <= alto_sax_min_hz] = alto_sax_min_hz
    freq[freq >= alto_sax_max_hz] = alto_sax_max_hz
    env = np.r_[env, np.zeros(note_len_samples)]

    #x = additive_synth_sawtooth(freq, env, sampling_rate, additive_synth_k=30)
    x = additive_synth_sawtooth(freq, env, sampling_rate, max_freq_hz = 16000)
    #--- in order to apply denoiser, we need the pytorch Tensor, so set return_numpy_arr to False
    x_hat = generate_from_audio(x, gen, return_numpy_arr = False)
    
    x_hat_den = denoiser(x_hat.squeeze(1), 4*denoising_strength)
    #x = x.numpy()[0]
    return x_hat, x_hat_den

#------- Globals ("consts") -----------
g_mel = MelSpec()
denoising_strength = 0.05
#--------------------------------------
    
if __name__ == '__main__':  
    #denoising_strength = 0.05
    gen_path = '../results/2023_05_28_hifigan_ssynth44khz_synthesized_input_16k_spl0.5_nonorm/hifigan_gen_checkpoint_3000.pt'
    gen, denoiser, train_cfg = load_generator_model(gen_path)    
    
    #--- define the MEL spectrum method
    #--- in hifigan code they use 2 implementations
    #--- (1) from fastpitch, when pre-calculating mel spec in prepare_dataset.sh. This is saved to disk and used for training/inference as mel spec INPUT 
    #--- (2) from hifigan, during training, when calculating mel spec for OUTPUT (target) signal, and for inference
    #--- NOTE use the same implementation that was used for INPUT in training. >>> starting 2023-05-28 and after, this should be 'hifigan' <<<
    #--- TODO put all this into a class -- ?
    MEL_IMPL = 'hifigan' #'fastpitch' #
    g_mel.init(train_cfg, MEL_IMPL)

    flist_path = '../data_ssynth/filelists' #
    mel_loss, mel_len, yret = run_on_validation_set(gen, denoiser, flist_path, return_file_index = 0)
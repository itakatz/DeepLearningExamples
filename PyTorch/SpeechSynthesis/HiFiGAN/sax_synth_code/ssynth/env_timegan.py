from utils import midi

import os
import glob
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import librosa
#from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, sosfiltfilt

import math
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

#--- I created symlink to the TimeGAN_pytorch_fork folder, so no need to add ~/git_repos to path
#import sys
#sys.path.append('/home/mlspeech/itamark/git_repos')


class gCFG():
    pd_cfg = dict(win = 1024, ac_win = 512, hop = 256)  
    range_hz = midi.MidiUtils.alto_sax_range_pmhs_hz  
    batch_size = 128
    sample_sec = 1 #2  #0.5
    history_len = 16 #32 # 6 # 32 frames with hop of 256 at 44100 Hz, is approx 18.5 msec of recent history
    env_db = True

class PositionalEncoding(nn.Module):
    ''' copied from here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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

class EnvelopesDataset(Dataset):
    def __init__(self, data_dir, filelist, env_params, sample_len_sec, history_len_samples = 1, cache = True, smoothing = False, seed = 1234, cache_dir = './feature_cache'):
        self.data_dir = data_dir #    data_dir = '../../data_ssynth'
        self.cache_dir = cache_dir #'./feature_cache'
        
        if cache:
            self.cache_flist = glob.glob(f'{self.cache_dir}/*')

        with open(filelist, 'r') as f:
            flist = f.readlines()
        #--- parse phrase ids from list of files
        ids_list = [fnm.rstrip().split('/')[-1].split('.wav')[0] for fnm in flist]
        phrase_df_fnm = f'{data_dir}/phrase_df.csv'
        phrase_df = pd.read_csv(phrase_df_fnm, index_col = 0).reset_index(drop = True)
        self.phrase_df = phrase_df[phrase_df['phrase_id'].isin(ids_list)]

        self.env_params = env_params
        self.sample_len_sec = sample_len_sec
        self.sample_len = None
        self.history_len_samples = history_len_samples
        self.cache = cache
        self.env_cache = {}
        self.smoothing = smoothing
        self.rng = np.random.default_rng(seed)
        self.sampling_rate = 44100 #--- hard code since we need to know it in case we don't load wavs at all (if reading features only from cache)
        filt_order = 2
        filt_cutoff_hz = 20
        self.lowpass_sos = butter(filt_order, filt_cutoff_hz, output = 'sos', fs = 44100 / env_params['hop_len_samples'])
        self.sample_start_index_randomly = True
    
    def sample_start_index(self, env_len):
        ''' for validation purpose, it is handy to be able to sample the same sub-sequence (just start from sample 0)
        '''
        if self.sample_start_index_randomly:
            return self.rng.integers(0, env_len - self.sample_len) # np.random.randint(0, env_len - self.sample_len)
        else:
            return 0

    def __getitem__(self, index):
        pinfo = self.phrase_df.iloc[index]
        cache_fnm = f'{self.cache_dir}/{pinfo.phrase_id}.pt'
        
        hop_len = self.env_params['hop_len_samples'] #256
        frame_len = self.env_params['frame_len_samples'] #512
        if self.sample_len is None:
            self.sample_len = int(self.sample_len_sec * self.sampling_rate / hop_len)

        if not self.cache or cache_fnm not in self.cache_flist:  #index not in self.env_cache:
            fnm = pinfo['file_nm']
            
            wav_fnm = f'{self.data_dir}/wavs/{pinfo.phrase_id}.wav'
            seg, sampling_rate = librosa.load(wav_fnm, sr = None)
            if sampling_rate != self.sampling_rate:
                raise ValueError(f'sampling rate is assumed to be {self.sampling_rate}, wav loaded has {sampling_rate}')
            
            file_id =  fnm.split('.')[0] # '01_Free_Improv_dynamic_mic'
            midi_fnm = f'{self.data_dir}/auto_midi/{file_id}.mid'
            #print(f'reading midi file {os.path.basename(midi_fnm)}')
            midi_df, midi_pitch, midi_aftertouch, midi_cc = midi.read_midi_to_df(midi_fnm)
            midi.verify_midi(midi_df)
            t0 = pinfo.sample_start / sampling_rate
            midi_p = midi.midi_phrase_from_dataframe(pinfo, midi_df, sampling_rate)

            #--- extract env
            env = librosa.feature.rms(y = seg, frame_length = frame_len, hop_length = hop_len, center = True)
            env = 1.3 * np.sqrt(2) * env[0]

            #--- extract pitch: use frame and window (auto-corr window) from pd_cfg, but use hop from the env_params so we get the same rate for env and pitch
            pitch, vflag, vprob = librosa.pyin(seg,
                                              fmin = gCFG.range_hz[0],
                                              fmax = gCFG.range_hz[1],
                                              sr = sampling_rate,
                                              frame_length = gCFG.pd_cfg['win'], 
                                              win_length = gCFG.pd_cfg['ac_win'],
                                              hop_length = hop_len,
                                              resolution = 0.05, #--- 5 cents pitch resolution
                                              center = True,
                                              max_transition_rate = 100)
            # times1 = librosa.times_like(f1, sr = sr, hop_length = hop)
            #no_note1 = (~vflag1)
            
            if self.smoothing:
                #--- spline smoothing turned out to be a bad idea, as it depends on signal scale and lenght (the min-squared-err threshold criterion is input to method)
                #t0 = 0.
                #ts = t0 + np.arange(0, len(env)) / sampling_rate
                #spl = UnivariateSpline(ts, env, s = self.spline_smoothing, k = 2)
                #env = spl(ts)
                env = sosfiltfilt(self.lowpass_sos, env)
                env[env < 0.] = 0.

            env = env.astype(np.float32)
            
            #--- add note features per frame: note-id, is-note, time-since-last-onset, time-since-last-offset
            midi_p = midi_p.reset_index(drop = True)
            note_id = np.zeros_like(env, dtype = np.int32)
            note_en = np.zeros_like(env, dtype = np.float32)
            is_note = np.zeros_like(env, dtype = np.float32)
            # TODO ! read params such as sample rate and frame hop, from config!
            frames_ind = np.round((midi_p['ts_sec'].to_numpy(dtype = float) - t0) * 44100 / 256).astype(int)
            #--- treat out-of-range frame index - TODO check why it happens, in method "midi.midi_phrase_from_dataframe"
            frames_ind = np.maximum(0, frames_ind)
            frames_ind = np.minimum(len(env), frames_ind)

            for k in range(0, len(midi_p), 2):
                i0, i1 = frames_ind[k : k + 2]
                note_en[i0 : i1] = np.median(env[i0:i1])
                note_id[i0 : i1] = midi_p.iloc[k].note
                is_note[i0 : i1] = 1.
        else:
            #env = self.env_cache[index]
            env, midi_p, t0, pitch, vprob, vflag, note_id, note_en, is_note = torch.load(cache_fnm)
        
        #--- save to cache
        if self.cache and cache_fnm not in self.cache_flist: #index not in self.env_cache:
            #self.env_cache[index] = env
            torch.save([env, midi_p, t0, pitch, vprob, vflag, note_id, note_en, is_note], cache_fnm)
            self.cache_flist.append(cache_fnm)

        #--- create a view which includes history TODO we can have different history_len for each feature
        if gCFG.env_db == True:
            noise_floor_db = -50. # empirically, that's the 0.05 quantile of smallest env value which is not 0, sampled over ~200 phrases
            env = 10 * np.log10(env + 10 ** (noise_floor_db / 10))
            #--- scale to [0, 1] (using lazy params, better to fit a min-max scaler over all features)
            env = env / (-noise_floor_db) + 1.

        env = sliding_window_view(env, self.history_len_samples)
        note_id = sliding_window_view(note_id, self.history_len_samples)
        note_en = sliding_window_view(note_en, self.history_len_samples)
        is_note = sliding_window_view(is_note, self.history_len_samples)
        
        #--- samplea sub-sequence of fixed length 
        env_len = env.shape[0]
        if env_len > self.sample_len:
            start_ind = self.sample_start_index(env_len) # self.rng.integers(0, env_len - self.sample_len) # np.random.randint(0, env_len - self.sample_len)
            env     =       env[start_ind : start_ind + self.sample_len].copy()
            note_en =   note_en[start_ind : start_ind + self.sample_len].copy()
            note_id =   note_id[start_ind : start_ind + self.sample_len].copy()
            is_note =   is_note[start_ind : start_ind + self.sample_len].copy()
        else:
            pad_sz = (self.sample_len - env_len, self.history_len_samples)
            env     = np.r_[env,     np.zeros(pad_sz, dtype = np.float32)]
            note_id = np.r_[note_id, np.zeros(pad_sz, dtype = np.int32)]
            note_en = np.r_[note_en, np.zeros(pad_sz, dtype = np.float32)]
            is_note = np.r_[is_note, np.zeros(pad_sz, dtype = np.float32)]

        #--- set input and output
        env_inp, env_out, env_len = env, env[:, -1:], env.shape[0]

        return env_inp, env_out, env_len, note_id, note_en, is_note
        #--- get envelope 
        #max_freq_hz = 16000
        #pd_cfg = dict(win = 1024, 
        #              ac_win = 512, # autocorrelation window
        #              hop = 256)

        #_x, _env, freq, raw_pitch = wav_midi_to_synth(seg, sampling_rate, midi_p, t0, pd_cfg, max_freq_hz, spline_smoothing = 2, verbose = False)
        #env = librosa.feature.rms(y = seg, frame_length = 512, hop_length = 256, center = True)
        #env = 1.3 * np.sqrt(2) * env[0] 
    
    def __len__(self):
        return self.phrase_df.shape[0]

def get_env_train_val_data(cache_dir, batch_size = 32, sample_dur_sec = 0.5, history_len = 1, workers = 0):
    env_params = dict(frame_len_samples = 512, hop_len_samples = 256)
    apply_smoothing = True
    home_dir = os.environ['HOME']
    data_dir = f'{home_dir}/ssynth/git_repos/DeepLearningExamples/PyTorch/SpeechSynthesis/HiFiGAN/data_ssynth'
    env_data_train = EnvelopesDataset(data_dir, '../../data_ssynth/filelists/ssynth_audio_train.txt', env_params, sample_dur_sec, history_len, smoothing = apply_smoothing, cache_dir = cache_dir)
    env_data_val = EnvelopesDataset(data_dir, '../../data_ssynth/filelists/ssynth_audio_val.txt', env_params, sample_dur_sec, history_len, smoothing = apply_smoothing, cache_dir = cache_dir)

    env_data_train_loader = DataLoader(env_data_train, batch_size, shuffle = True, num_workers = workers)
    env_data_val_loader = DataLoader(env_data_val, batch_size, shuffle = False, num_workers = workers)

    return env_data_train_loader, env_data_val_loader

def save_features_to_cache(cache_dir, n_workers = 32):
    ''' just read all samples from the dataset class, so features are saved to cache
    '''
    from multiprocessing.pool import Pool
    train_loader, val_loader = get_env_train_val_data(cache_dir, gCFG.batch_size, gCFG.sample_sec, gCFG.history_len)

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
    input_args = f'env_timegan.py --calc_z_grad --num_layer 3 --num_layer_gen 3 --num_layer_discrim 3 --hidden_dim 64 --latent_dim 8 --embedding_dim 32 --batch_size {gCFG.batch_size} --outf results/2024_04_04 --model EnvelopeTimeGAN --name lyr_3_ldim_8'
    sys.argv = input_args.split()
    opt = Options().parse()
    
    #--- data loaders
    cache_dir = 'feature_cache2'
    workers = 0 #opt.workers # using more then 1 process for loading just makes it slower (overhead of data to/from sub-process?)
    train_loader, val_loader = get_env_train_val_data(cache_dir, gCFG.batch_size, gCFG.sample_sec, gCFG.history_len, workers)
    
    #--- different no. of epoch for embed/supervised and for joint training
    opt.num_epochs_es = 50 #250 #opt.iteration
    opt.num_epochs = 1000 # opt.iteration
    #opt.batch_size = gCFG.batch_size

    x, xout, t, note_id, note_en, is_note = train_loader.dataset[0] #--- get a sample for the dims
    opt.seq_len = x.shape[0] #167 # 86 - history_len + 1 #344*2+1
    opt.z_dim = x.shape[1] #history_len #1 # number of features per sequence frame
    opt.z_dim_out = xout.shape[1] #1

    #opt.batch_size = batch_size
    #opt.module = 'gru'
    #opt.outf = './output_TMP'
    opt.average_seq_before_loss = True
    opt.generator_loss_moments_axis = 1 # use "0" to calculate along batch (original impl, after bug fix), or "1" to calculate along the sequence (makes more sense)
    #--- load autoencode and supervisor (aka AES) from disk to start from joint training
    AES_checkpoint = './results/2023_30_12_ldim2/test1_gen_latent_dim2/train/weights'
    AES_epoch = 249
    joint_train_only = True

    #--- to load model:
    #opt.resume = 'results/2023_17_12_test/test4_mean_bce/train/weights'
    #opt.resume_epoch = 0

    # opt.resume='results/2023_31_12_ldim2/bug_fix_and_smooth_loss1/train/weights'
    # opt.resume_epoch=499

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

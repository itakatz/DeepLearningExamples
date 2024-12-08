import os
import glob
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt
import librosa

import torch
import torchaudio
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from ssynth.utils import midi

class gCFG():
    pd_cfg = dict(win = 1024, ac_win = 512, hop = 256)  
    range_hz = midi.MidiUtils.alto_sax_range_pmhs_hz  
    batch_size = 128
    sample_sec = 1 #2  #0.5
    history_len = 16 #32 # 6 # 32 frames with hop of 256 at 44100 Hz, is approx 185 msec of recent history
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

def sliding_window(arr, win_size, dim = 0, step_size = 1, mode = 'numpy'):
    if mode == 'numpy':
        return sliding_window_view(arr, win_size, axis = dim)
    elif mode == 'torch':
        return arr.unfold(dim, win_size, step_size)
    else:
        raise ValueError('mode shuld be "numpy" or "torch"')

def copy_or_clone(arr, mode):
    __DEBUG_DISABLE_TORCH_CLONE = True
    if mode == 'numpy':
        return arr.copy()
    elif mode == 'torch':
        #--- TODO check if torch can handle the view itself (return arr instead of arr.clone())
        if __DEBUG_DISABLE_TORCH_CLONE:
            return arr
        else:
            return arr.clone()
    else:
        raise ValueError('mode shuld be "numpy" or "torch"')

def zero_pad_tail(arr, pad_sz, mode):
    if mode == 'numpy':
        dtype = arr.dtype
        dim1 = arr.shape[1]
        return np.r_[arr, np.zeros((pad_sz, dim1), dtype = dtype)]
    elif mode == 'torch':
        try:
            return F.pad(arr, (0, 0, 0, pad_sz)) #--- F.pad accepts tuple of (start_dim0, end_dim0, start_dim1, end_dim1) values of pad size
        except TypeError as e:
            print(f'error: {e}, input shape: {arr.shape}, pad_sz: {pad_sz}')
            raise e
    else:
        raise ValueError('mode shuld be "numpy" or "torch"')

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
        self.sampling_rate = 44100 #--- hard code since we need to know it in case we don't load wavs at all (if reading features only from cache)
        self.sample_len_sec = sample_len_sec
        #self.sample_len = None
        hop_len = self.env_params['hop_len_samples'] #256
        self.sample_len = int(self.sample_len_sec * self.sampling_rate / hop_len)
        self.history_len_samples = history_len_samples
        self.cache = cache
        self.env_cache = {}
        self.smoothing = smoothing
        self.rng = np.random.default_rng(seed)
        filt_order = 2
        filt_cutoff_hz = 20
        self.lowpass_sos = butter(filt_order, filt_cutoff_hz, output = 'sos', fs = self.sampling_rate / env_params['hop_len_samples'])
        self.sample_start_index_randomly = True
    
    def sample_start_index(self, env_len):
        ''' for validation purpose, it is handy to be able to sample the same sub-sequence (just start from sample 0)
        '''
        if self.sample_start_index_randomly:
            random_idx = self.rng.integers(0, env_len - self.sample_len + 1) # np.random.randint(0, env_len - self.sample_len)
        else:
            random_idx = 0
        return random_idx

    def __getitem__(self, index):
        ''' NOTE (1) the features created here (pitch and env) are slightly different from the featured created and saved in "sax_synth_code/prepare_synthetic.py":
            1. The pitch here is "raw", and in saved features it is corrected/enhanved using the midi data (closing gaps etc.)
            2. The env here is created in sampling rate of 1:256 (hop_length) compared to audio, while in the saved features it is first created in audio rate,
               then down-sampled. 
            see also the method ssynth.utils.synthesis.wav_midi_to_synth
            NOTE (2) therefore, I comment out the feature-creation (env and pitch) here and just load it from disk. It still makes sense to save again to feature-cache,
                     since we also cache note_en, note_id, and is_note
        '''
        pinfo = self.phrase_df.iloc[index]
        cache_fnm = f'{self.cache_dir}/{pinfo.phrase_id}.pt'
        
        hop_len = self.env_params['hop_len_samples'] #256
        frame_len = self.env_params['frame_len_samples'] #512

        if not self.cache or cache_fnm not in self.cache_flist:  #index not in self.env_cache:
            fnm = pinfo['file_nm']
            
            file_id =  fnm.split('.')[0] # '01_Free_Improv_dynamic_mic'
            midi_fnm = f'{self.data_dir}/auto_midi/{file_id}.mid'
            #print(f'reading midi file {os.path.basename(midi_fnm)}')
            midi_df, midi_pitch, midi_aftertouch, midi_cc = midi.read_midi_to_df(midi_fnm)
            midi.verify_midi(midi_df)
            t0 = pinfo.sample_start / self.sampling_rate
            midi_p = midi.midi_phrase_from_dataframe(pinfo, midi_df, self.sampling_rate)

            if True:
                tmode = 'torch'
                env_fnm = f'{self.data_dir}/env_synth__lp20hz/{pinfo.phrase_id}.pt'
                pitch_fnm = f'{self.data_dir}/pitch_synth/{pinfo.phrase_id}.pt'
                env = torch.load(env_fnm)[0]
                pitch_dict = torch.load(pitch_fnm)
                pitch, vflag, vprob = pitch_dict['freq'][0], pitch_dict['vflag'][0], pitch_dict['vprob'][0]
            else: #--- old code: extract pitch and env
                #--- extract env
                tmode = 'numpy'
                wav_fnm = f'{self.data_dir}/wavs/{pinfo.phrase_id}.wav'
                seg, sampling_rate = librosa.load(wav_fnm, sr = None)
                if sampling_rate != self.sampling_rate:
                    raise ValueError(f'sampling rate is assumed to be {self.sampling_rate}, wav loaded has {sampling_rate}')
                env = librosa.feature.rms(y = seg, frame_length = frame_len, hop_length = hop_len, center = True)
                env = 1.3 * np.sqrt(2) * env[0]
                
                if self.smoothing:
                    env = sosfiltfilt(self.lowpass_sos, env)
                    env[env < 0.] = 0.

                env = env.astype(np.float32)

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
            
            #--- add note features per frame: note-id, is-note, time-since-last-onset, time-since-last-offset
            midi_p = midi_p.reset_index(drop = True)
            note_id = np.zeros_like(env, dtype = np.int64)
            note_en = np.zeros_like(env, dtype = np.float32)
            is_note = np.zeros_like(env, dtype = np.float32)
            # TODO ! read params such as sample rate and frame hop, from config!
            frames_ind = np.round((midi_p['ts_sec'].to_numpy(dtype = float) - t0) * self.sampling_rate / hop_len).astype(int)
            #--- treat out-of-range frame index - TODO check why it happens, in method "midi.midi_phrase_from_dataframe"
            frames_ind = np.maximum(0, frames_ind)
            frames_ind = np.minimum(len(env), frames_ind)

            for k in range(0, len(midi_p), 2):
                i0, i1 = frames_ind[k : k + 2]
                note_en[i0 : i1] = np.median(env[i0:i1])
                note_id[i0 : i1] = midi_p.iloc[k].note
                is_note[i0 : i1] = 1.
            if tmode == 'torch':
                #note_en = torch.Tensor(note_en, dtype = torch.int64)
                #note_id = torch.Tensor(note_id, dtype = torch.int64)
                #is_note = torch.Tensor(is_note, dtype = torch.int64)
                #--- try to use a tensor with memory shared with the numpy array
                note_en = torch.from_numpy(note_en)
                note_id = torch.from_numpy(note_id)
                is_note = torch.from_numpy(is_note)

        else:
            #env = self.env_cache[index]
            env, midi_p, t0, pitch, vprob, vflag, note_id, note_en, is_note = torch.load(cache_fnm, weights_only = False)
            tmode = 'numpy' if type(env) is np.ndarray else 'torch'
        
        #--- save to cache
        if self.cache and cache_fnm not in self.cache_flist: #index not in self.env_cache:
            #self.env_cache[index] = env
            torch.save([env, midi_p, t0, pitch, vprob, vflag, note_id, note_en, is_note], cache_fnm)
            self.cache_flist.append(cache_fnm)

        #--- create a view which includes history TODO we can have different history_len for each feature
        if gCFG.env_db == True:
            noise_floor_db = -50. # empirically, that's the 0.05 quantile of smallest env value which is not 0, sampled over ~200 phrases
            if tmode == 'numpy':
                env = 10 * np.log10(env + 10 ** (noise_floor_db / 10))
            elif tmode == 'torch':
                env = 10 * torch.log10(env + 10 ** (noise_floor_db / 10))
            else:
                raise ValueError
            #--- scale to [0, 1] (using lazy params, better to fit a min-max scaler over all features)
            env = env / (-noise_floor_db) + 1.

        env = sliding_window(env, self.history_len_samples, mode = tmode)
        pitch = sliding_window(pitch, self.history_len_samples, mode = tmode)
        #___ TODO continue impl with "tmode", for note_* they are currently numpy array so...
        note_id = sliding_window(note_id, self.history_len_samples, mode = tmode)
        note_en = sliding_window(note_en, self.history_len_samples, mode = tmode)
        is_note = sliding_window(is_note, self.history_len_samples, mode = tmode)
        
        #--- samplea sub-sequence of fixed length 
        env_len = env.shape[0]
        if env_len > self.sample_len:            
            start_ind = self.sample_start_index(env_len)
            self._start_idx = start_ind
            env     =   copy_or_clone(env[start_ind : start_ind + self.sample_len], tmode)
            note_en =   copy_or_clone(note_en[start_ind : start_ind + self.sample_len], tmode)
            note_id =   copy_or_clone(note_id[start_ind : start_ind + self.sample_len], tmode)
            is_note =   copy_or_clone(is_note[start_ind : start_ind + self.sample_len], tmode)
            pitch   =   copy_or_clone(pitch[start_ind : start_ind + self.sample_len], tmode)
        else:
            #--- TODO it's probably better to pad and return a mask as well
            self._start_idx = 0
            pad_sz = self.sample_len - env_len #(self.sample_len - env_len, self.history_len_samples)
            env     = zero_pad_tail(env, pad_sz, tmode)
            note_id = zero_pad_tail(note_id, pad_sz, tmode)
            note_en = zero_pad_tail(note_en, pad_sz, tmode)
            is_note = zero_pad_tail(is_note, pad_sz, tmode)
            pitch   = zero_pad_tail(pitch, pad_sz, tmode)

        #--- set input and output
        env_inp, env_out, env_len = env, env[:, -1:], env.shape[0]

        return env_inp, env_out, env_len, note_id, note_en, is_note, pitch, index
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

class PairedAudioDataset(Dataset):
    def __init__(self, data_dir, filelist, env_params, sample_len_sec, cache_dir, fs = 16000):
        ''' in DiffAR impl, this class holds 2 identical instances of the Dataset class. But there is no need, 
            we can simply copy the audio and use it as 'audio_b'
        '''
        self.data_dir = data_dir
        self.env_params = env_params
        self.fs = fs

        self.ds = EnvelopesDataset(
            data_dir,
            filelist,
            env_params,
            sample_len_sec,
            history_len_samples = 1,
            cache = True,
            smoothing = True,
            cache_dir = cache_dir
        )
       
        #self.ds_b = EnvelopesDataset(
        #    data_dir,
        #   filelist,
        #    env_params,
        #    sample_len_sec,
        #    history_len_samples = 1,
        #    cache = True,
        #    smoothing = True,
        #    cache_dir = cache_dir
        #)
        
        self.n_samples = self.ds.sample_len
        
        self.alto_sax_range_midi = librosa.note_to_midi(midi.MidiUtils.alto_sax_range_pmhs_notes)
        #assert len(self.ds_a) == len(self.ds_b), "datasets in `PairedAudioDataset` must be of equal length"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        # Using the same index for iterators.
        
        env_inp, env_out, env_len, note_id, note_en, is_note, pitch, index = self.ds[ind]
        
        #--- load audio
        phrase = self.ds.phrase_df.iloc[ind]
        wav_dir = 'wavs' if self.fs == 44100 else 'wavs_16khz'
        wav_fnm = f'{self.data_dir}/{wav_dir}/{phrase.phrase_id}.wav'
        audio, fs = torchaudio.load(wav_fnm)
        assert fs == self.fs #16000 
        fs_orig = 44100 #--- envelopes were calculated at 44.1khz, so we need to convert indices to original audio at 16khz
        start_index = self.ds._start_idx
        #--- convert frame-rate to audio-rate
        hop_sz = self.env_params['hop_len_samples']
        win_sz = self.env_params['frame_len_samples']
        #--- window centers at the oroginal fs (44.1khz)
        audio_len = int((env_len - 1) * hop_sz / fs_orig * fs) #--- must be a constant, otheriwise DataLoader will recieve elements of different lenght in a batch
        t1, t2 = start_index * hop_sz / fs_orig, (start_index + env_len - 1) * hop_sz / fs_orig
        n1, _ = int(t1 * fs) , int(t2 * fs)
        #--- override n2 to make sure it is a constant within a batch (before this fix, DataLoader threw exception becuase of different lengths)
        n2 = n1 + audio_len
        audio = audio[:, n1 : n2]
        N = audio.shape[1]

        #--- interpolate features to audio rate
        note_en_ar = torch.nn.functional.interpolate(note_en.view(1,1,-1), size = N, mode = 'nearest-exact').squeeze(0)
        note_ind = note_id > 0
        range_min = self.alto_sax_range_midi[0]
        range_max = self.alto_sax_range_midi[-1]
        note_id = note_id.to(torch.float32)
        note_id[note_ind] = (note_id[note_ind] - range_min + 1) / (range_max - range_min + 1)
        is_note_ar = torch.nn.functional.interpolate(is_note.view(1,1,-1), size = N, mode = 'nearest-exact').squeeze(0)
        note_id_ar = torch.nn.functional.interpolate(note_id.view(1,1,-1), size = N, mode = 'nearest-exact').squeeze(0)

        overlap_zone = N // 3 # hard coded, a third of winfow len
        audio_context = audio.detach().clone()
        audio_context[:, overlap_zone:].data.zero_()

        note_context = torch.concat([note_id_ar,is_note_ar])

        return audio, audio_context, note_context, note_en_ar, overlap_zone

def get_env_train_val_data(cache_dir, batch_size = 32, sample_dur_sec = 0.5, history_len = 1, workers = 0, shuffle_train_loader = True, train_drop_last = False):
    env_params = dict(frame_len_samples = 512, hop_len_samples = 256)
    apply_smoothing = True
    home_dir = os.environ['HOME']
    data_dir = f'{home_dir}/ssynth/git_repos/DeepLearningExamples/PyTorch/SpeechSynthesis/HiFiGAN/data_ssynth'
    env_data_train = EnvelopesDataset(data_dir, f'{data_dir}/filelists/ssynth_audio_train.txt', env_params, sample_dur_sec, history_len, smoothing = apply_smoothing, cache_dir = cache_dir)
    env_data_val = EnvelopesDataset(data_dir, f'{data_dir}/filelists/ssynth_audio_val.txt', env_params, sample_dur_sec, history_len, smoothing = apply_smoothing, cache_dir = cache_dir)

    env_data_train_loader = DataLoader(env_data_train, batch_size, shuffle = shuffle_train_loader, num_workers = workers, drop_last = train_drop_last)
    env_data_val_loader = DataLoader(env_data_val, batch_size, shuffle = False, num_workers = workers)

    return env_data_train_loader, env_data_val_loader

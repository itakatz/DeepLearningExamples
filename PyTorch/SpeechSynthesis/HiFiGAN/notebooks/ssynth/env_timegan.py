from utils import midi

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import librosa
#from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, sosfiltfilt

from torch.utils.data import Dataset, DataLoader

#--- I created symlink to the TimeGAN_pytorch_fork folder, so no need to add ~/git_repos to path
#import sys
#sys.path.append('/home/mlspeech/itamark/git_repos')

pd_cfg = dict(win = 1024, ac_win = 512, hop = 256)  
range_hz = midi.MidiUtils.alto_sax_range_pmhs_hz  

class EnvelopesDataset(Dataset):
    def __init__(self, data_dir, filelist, env_params, sample_len_sec, history_len_samples = 1, cache = True, smoothing = False, seed = 1234):
        self.data_dir = data_dir #    data_dir = '../../data_ssynth'
        
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
        if not self.cache or index not in self.env_cache:
            pinfo = self.phrase_df.iloc[index]
            wav_fnm = pinfo['file_nm']
            
            if False:
                #--- TODO cache (pre-save to disk, or while loading for the first time)
                file_id =  wav_fnm.split('.')[0] # '01_Free_Improv_dynamic_mic'
                midi_fnm = f'{self.data_dir}/auto_midi/{file_id}.mid'
                #print(f'reading midi file {os.path.basename(midi_fnm)}')
                midi_df, midi_pitch, midi_aftertouch, midi_cc = midi.read_midi_to_df(midi_fnm)
                midi.verify_midi(midi_df)
                t0 = pinfo.sample_start / sampling_rate
                midi_p = midi.midi_phrase_from_dataframe(pinfo, midi_df, sampling_rate)

            #phrase_df = phrase_df[phrase_df.file_nm.str.contains(file_id)]
            #phrase_ind = 14 #26
            #p = phrase_df.iloc[phrase_ind]

            wav_fnm = f'{self.data_dir}/wavs/{pinfo.phrase_id}.wav'
            #seg, sr = librosa.load(wav_fnm, sr = sampling_rate)
            seg, sampling_rate = librosa.load(wav_fnm, sr = None)
            hop_len = self.env_params['hop_len_samples'] #256
            frame_len = self.env_params['frame_len_samples'] #256

            if self.sample_len is None:
                self.sample_len = int(self.sample_len_sec * sampling_rate / hop_len)
            
            #--- extract env
            env = librosa.feature.rms(y = seg, frame_length = frame_len, hop_length = hop_len, center = True)
            env = 1.3 * np.sqrt(2) * env[0]

            #--- extract pitch: use frame and window (auto-corr window) from pd_cfg, but use hop from the env_params so we get the same rate for env and pitch
            f1, vflag1, vprob1 = librosa.pyin(seg,
                                              fmin = range_hz[0],
                                              fmax = range_hz[1],
                                              sr = sampling_rate,
                                              frame_length = pd_cfg['win'], 
                                              win_length = pd_cfg['ac_win'],
                                              hop_length = hop_len,
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
        else:
            env = self.env_cache[index]

        if self.cache and index not in self.env_cache:
            self.env_cache[index] = env

        env_len = env.shape[0]
        if env_len > self.sample_len:
            start_ind = self.sample_start_index(env_len) # self.rng.integers(0, env_len - self.sample_len) # np.random.randint(0, env_len - self.sample_len)
            env = env[start_ind : start_ind + self.sample_len]
        else:
            env = np.r_[env, np.zeros(self.sample_len - env_len, dtype = np.float32)]
        
        #env = env[:, np.newaxis]
        env = sliding_window_view(env, self.history_len_samples).copy()

        #--- set input and output
        inp, out, len = env, env[:, -1:], env.shape[0]
        return inp, out, len
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

def get_env_train_val_data(batch_size = 32, sample_dur_sec = 0.5, history_len = 1):
    env_params = dict(frame_len_samples = 512, hop_len_samples = 256)
    apply_smoothing = True
    data_dir = '/home/mlspeech/itamark/ssynth/git_repos/DeepLearningExamples/PyTorch/SpeechSynthesis/HiFiGAN/data_ssynth'
    env_data_train = EnvelopesDataset(data_dir, '../../data_ssynth/filelists/ssynth_audio_train.txt', env_params, sample_dur_sec, history_len, smoothing = apply_smoothing)
    env_data_val = EnvelopesDataset(data_dir, '../../data_ssynth/filelists/ssynth_audio_val.txt', env_params, sample_dur_sec, history_len, smoothing = apply_smoothing)

    env_data_train_loader = DataLoader(env_data_train, batch_size, shuffle = True)
    env_data_val_loader = DataLoader(env_data_val, batch_size, shuffle = False)

    return env_data_train_loader, env_data_val_loader

if __name__ == '__main__':
    from TimeGAN_pytorch_fork.options import Options
    from TimeGAN_pytorch_fork.lib.env_timegan import EnvelopeTimeGAN
    batch_size = 64
    sample_sec = 1 #0.5
    history_len = 6
    train_loader, val_loader = get_env_train_val_data(batch_size, sample_sec, history_len)

    opt = Options().parse()
    #--- different no. of epoch for embed/supervised and for joint training
    opt.num_epochs_es = 250 #opt.iteration
    opt.num_epochs = opt.iteration
    

    x, xout, t = train_loader.dataset[0] #--- get a sample for the dims
    opt.seq_len = x.shape[0] #167 # 86 - history_len + 1 #344*2+1
    opt.z_dim = x.shape[1] #history_len #1 # number of features per sequence frame
    opt.z_dim_out = xout.shape[1] #1
    
    #opt.batch_size = batch_size
    #opt.module = 'gru'
    #opt.outf = './output_TMP'
    model = EnvelopeTimeGAN(opt, train_loader, val_loader)
    model.max_seq_len = opt.seq_len
    print(f'Options:\n{opt}')
    model.train()

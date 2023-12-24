import librosa
import numpy as np
from scipy.signal import medfilt
import glob

class PhraseSegmentCfg():
    def __init__(self, sr):
        self.sr = sr
        self.noise_floor_quantile = 0.01
        self.phrase_min_dur_sec = 1
        self.phrase_max_dur_sec = 15
        self.noise_gap_db = 6 #--- when signal envelope is less then this value above the noise floor, consider it 'silence'
        self.phrase_pause_sec = 0.25 #--- should be in 'beats' units, e.g 2 quarters. TODO
        self.phrase_pause_short_sec = 0.125 #--- for searching short pauses in case regular ones were not found
        self.win_len_msec = 9.98 #--- to get 440 samples in a window and not 441 for 10 msec window
        self.win_len_samples = int(self.win_len_msec * 1e-3 * sr)
        self.win_len_samples += (self.win_len_samples % 2) #--- make sure it's even
        self.win_hop_samples = self.win_len_samples // 2

#--- for each long sengment, re-run detection with smaller pause threshold. also, local noise-floor estimation will be more useful
def split_long_phrases(y, st, en, cfg, verbose = False):
    seg_dur_sec = (st[1:] - en[:-1]) / cfg.sr
    ind_too_long = (seg_dur_sec > cfg.phrase_max_dur_sec).nonzero()[0]
    print(f'found {len(ind_too_long)} segments longer than {cfg.phrase_max_dur_sec} sec')
    for iind, ind in enumerate(ind_too_long[::-1]):
        #if iind==0:
        #    break
        istart = en[ind]
        iend = st[ind+1] if ind + 1 < len(st) else len(y)
        iseg = y[istart:iend]
        iseg_st, iseg_en, _ = get_audio_pauses(iseg, cfg,  short_pause = True, verbose = False)
        #--- correct shift and insert to global list
        iseg_st += istart
        iseg_en += istart
        #--- we can always insert in arbitrary order and sort at the end, so this is just a sanity check
        assert(np.all((iseg_st <  st[ind+1]) & ((iseg_st > st[ind]))))
        assert(np.all((iseg_en <  en[ind+1]) & ((iseg_en > en[ind]))))
        st = np.insert(st, ind + 1, iseg_st)
        en = np.insert(en, ind + 1, iseg_en)

        if verbose:
            print(f'[{iind}] seg {ind} with duration {seg_dur_sec[ind]:.1f}')
            print(f'\tfound {iseg_st.shape[0]} pauses')
            iseg_dur_sec = (iseg_st[1:] - iseg_en[:-1]) / cfg.sr
            print(f'\tnew durations of mid-segments (omitting 1st and last): {iseg_dur_sec.round(2)} sec')
        if False:
            #%matplotlib inline
            fig, ax = plt.subplots(figsize = (12,8))
            n1 = 0
            n2 = len(iseg_st)
            #ax.plot(np.arange(iseg_st[n1], iseg_en[n2]) / sr, iseg[iseg_st[n1] : iseg_en[n2]])
            ax.plot(np.arange(istart,iend) / sr, iseg)
            for k in range(n1,n2):
                ax.plot(np.arange(iseg_st[k], iseg_en[k]) / sr, iseg[iseg_st[k]-istart : iseg_en[k]-istart],'r:')
            #ax.set_xlim([11,13])

    #--- calc updated durations
    assert((en > st).all() and (np.diff(st) > 0).all() and (np.diff(en) > 0).all())
    return st, en

#--- treat too short segments
def merge_short_phrases(st, en, cfg):
    seg_dur_sec = (st[1:] - en[:-1]) / cfg.sr
    ind_too_short = (seg_dur_sec < cfg.phrase_min_dur_sec).nonzero()[0]
    print(f'found {len(ind_too_short)} segments shorter than {cfg.phrase_min_dur_sec} sec')
    ind_to_remove = []
    for ind in ind_too_short[::-1]:
        #print(st[ind+1]-en[ind])
        if ind + 1 < len(seg_dur_sec) and seg_dur_sec[ind - 1] < seg_dur_sec[ind + 1]:
            st = np.delete(st, ind)
            en = np.delete(en, ind)
        else:
            st = np.delete(st, ind + 1)
            en = np.delete(en, ind + 1)
        #print(seg_dur_sec[ind-1:ind+2])
    return st, en

def get_audio_pauses(y, cfg, short_pause = False, verbose = 0):
    #--- estimate noise floor in dB
    y = y.copy()
    y[y == 0] = np.nan #--- for finding the noise floor, don't take 0 values (it comes from space between recordings)

    env = librosa.feature.rms(y = y , frame_length = cfg.win_len_samples, hop_length = cfg.win_hop_samples)
    env = env[0]
    ts_samples = (np.arange(env.shape[0]) + 1) * cfg.win_hop_samples # / cfg.sr
    env_db = 10 * np.log10(env)
    noise_floor_db = np.nanquantile(env_db, cfg.noise_floor_quantile)
    noise_th = noise_floor_db + cfg.noise_gap_db
    
    env_db[np.isnan(env_db)] = noise_floor_db #--- set the 0 samples' envelope to the noise floor (avoid the nans)
    
    #--- apply median filter with length of the pause we search
    pause_sec = cfg.phrase_pause_short_sec if short_pause else cfg.phrase_pause_sec
    med_filt_len = int(cfg.sr * pause_sec / cfg.win_hop_samples)
    med_filt_len += (1 - (med_filt_len % 2))
    env_db_mf = medfilt(env_db, med_filt_len)
    
    if verbose:
        print(f'noise floor: {noise_floor_db:.1f}dB, noise th: {noise_th:.1f}dB, median filter len: {med_filt_len}')
        
    #--- pauses starts and ends
    st = np.where(np.diff((env_db_mf <= noise_th).astype(int)) == 1)[0]
    en = np.where(np.diff((env_db_mf <= noise_th).astype(int)) == -1)[0]
    
    #--- treat the case of env above th in start or end of envelope (end wo start at the beggining, or start wo end at the end)
    if len(st) == len(en) + 1:
        en = np.r_[en, len(env_db_mf) - 2] #--- add index to window before last, to make sure its ts_samples is not > len(y)
    elif len(en) == len(st) + 1:
        st = np.r_[0, st]

    #--- this fails if one of [st, en] is empty (that's why I added the if above)
    if(en[0] < st[0]):
        st = np.r_[0, st]   
 
    if(en[-1] < st[-1]):
        #st = st[:-1] # BUG: this omits last phrase
        en = np.r_[en, len(env_db_mf) - 2] #--- add index to window before last, to make sure its ts_samples is not > len(y)
        
    assert(len(st) == len(en))

    return ts_samples[st], ts_samples[en], env_db_mf

if __name__ == '__main__':
    #--- params
    data_folder = '/home/itamar/ssynth/data/wavs'
    flist = glob.glob(f'{data_folder}/*dynamic_mic.wav') 
    file_nm = flist[1]
    y, sr = librosa.load(file_nm, sr = None) #--- NOTE librosa converts 24 bit audio (which is int32) to float in [-1, 1]
    cfg = PhraseSegmentCfg(sr)


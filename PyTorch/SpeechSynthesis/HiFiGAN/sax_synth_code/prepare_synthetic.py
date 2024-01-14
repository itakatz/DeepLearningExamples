import os
import glob
import librosa
import torch
import numpy as np
import pandas as pd
import soundfile as sf

from ssynth.utils.midi import read_midi_to_df, verify_midi, midi_phrase_from_dataframe
from ssynth.utils.synthesis import wav_midi_to_synth

if __name__ == '__main__':
    #--- input params TODO use argparse
    #--- exactly one of these should be None:
    num_harmonics, max_freq_hz = np.inf, None # OR: None, 16000
    #num_harmonics, max_freq_hz = None, 16000 # OR: None, 16000
    assert num_harmonics is None or max_freq_hz is None
    smoothing_method = 'lowpass' # options are ['lowpass', 'spline', 'none']. NOTE  old method 'spline' is bad, amoothing is done using MSE criterion which depends on signal length
    smoothing_cfgs = dict(spline  = dict(method = 'spline',  smoothing = 0.5, order = 2), 
                          lowpass = dict(method = 'lowpass', cutoff_hz = 20, order = 2))
    sr = 44100
    pd_cfg = dict(win = 1024,
                  ac_win = 512, # autocorrelation window
                  hop = 256)
    
    data_folder = '/home/mlspeech/itamark/ssynth/git_repos/DeepLearningExamples/PyTorch/SpeechSynthesis/HiFiGAN/data_ssynth' # /wavs_raw'
    wavs_raw_folder = f'{data_folder}/wavs_raw'
    wavs_folder = f'{data_folder}/wavs'
    midi_folder = f'{data_folder}/auto_midi'
    flist = glob.glob(f'{wavs_raw_folder}/*Free*dynamic_mic*.wav')
    flist.sort()
    phrase_df = pd.read_csv(f'{data_folder}/phrase_df.csv', index_col = 0)
    
    suffix = f'{num_harmonics}h' if max_freq_hz is None else f'{int(max_freq_hz / 1000)}k'
    if smoothing_method == 'none':
        env_suffix = ''
    elif smoothing_method == 'lowpass':
        cutoff = smoothing_cfgs['lowpass']['cutoff_hz']
        env_suffix = f'_lp{cutoff}hz'
        suffix += env_suffix
    elif smoothing_method == 'spline':
        s = smoothing_cfgs['spline']['smoothing']
        env_suffix = f'_spl{s}'
        suffix += env_suffix
    else:
        raise ValueError(f'unknown smoothing method {smoothing_method}')
    
    synth_out_dir = wavs_folder.replace('/wavs', f'/wavs_synth_{suffix}')
    pitch_out_dir = wavs_folder.replace('/wavs', '/pitch_synth')
    env_out_dir   = wavs_folder.replace('/wavs', f'/env_synth_{env_suffix}')
    print(f'writing synthesized wavs to {synth_out_dir}')
    print(f'writing extracted pitch to {pitch_out_dir}')
    print(f'writing extracted envelopes to {env_out_dir}')
    
    if not os.path.isdir(synth_out_dir):
        os.mkdir(synth_out_dir)
    if not os.path.isdir(pitch_out_dir):
        os.mkdir(pitch_out_dir)
    if not os.path.isdir(env_out_dir):
        os.mkdir(env_out_dir)
    
    #--- iterate over files, and over phrases in an inner loop 
    for ifnm, fnm in enumerate(flist):
        #if ifnm < 2:
        #    continue
        fnm_base = os.path.basename(fnm)
        midi_fnm = fnm.replace('/wavs_raw/', '/auto_midi/').replace('.wav', '.mid')
        #if TEST_MODE and '_dynamic_mic' in midi_fnm:
        #    midi_fnm = midi_fnm.replace('_dynamic_mic', '')
            
        print(f'[{ifnm}] reading midi file {os.path.basename(midi_fnm)}')
        midi_df, midi_pitch, midi_aftertouch, midi_cc = read_midi_to_df(midi_fnm)
        verify_midi(midi_df)
        p_df = phrase_df.query("file_nm == @fnm_base").reset_index(drop = True)
        print(f'processing {p_df.shape[0]} phrases from file {fnm_base}')
        for iphrs, phrs in p_df.iterrows():
            wav_fnm = f'{wavs_folder}/{phrs.phrase_id}.wav'
            print(f'[{iphrs}] phrase {wav_fnm}')
            
            seg, sr = librosa.load(wav_fnm, sr = sr)
            midi_p = midi_phrase_from_dataframe(phrs, midi_df, sr)
            t0 = phrs.sample_start / sr
            try:
                #wav_midi_to_synth(seg, sr, midi_p, t0, pd_cfg, num_harmonics, max_freq_hz, spline_smoothing = None, verbose = False)
                seg_synth, env, pitch_dict = wav_midi_to_synth(seg, sr, midi_p, t0, pd_cfg, num_harmonics, max_freq_hz, smoothing_cfgs[smoothing_method], verbose = False)
                pitch = pitch_dict['freq']

            except Exception as e:
                print(f'phrase {iphrs} failed with error: {e}')
                continue
            #--- save synth signal and pitch
            fnm_out = f'{synth_out_dir}/{phrs.phrase_id}.wav'
            sf.write(fnm_out, seg_synth, sr, subtype = 'PCM_24')

            #--- save env and pitch
            pitch_fnm_out = f'{pitch_out_dir}/{phrs.phrase_id}.pt'
            if not os.path.isfile(pitch_fnm_out):
                pitch = torch.tensor(pitch[np.newaxis,:].astype(np.float32))
                torch.save(pitch, pitch_fnm_out)
            
            env_fnm_out = f'{env_out_dir}/{phrs.phrase_id}.pt'
            if not os.path.isfile(env_fnm_out):
                env = torch.tensor(env[np.newaxis,:].astype(np.float32))
                torch.save(env, env_fnm_out)
            #break
        #break

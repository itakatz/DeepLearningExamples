import soundfile as sf
#import IPython.display as ipd
import numpy as np
import pandas as pd
import glob
import os
import sys
import matplotlib
import mido
from mido import MidiFile
from scipy.interpolate import interp1d
import torch

import matplotlib.pyplot as plt
import librosa.display

from phrase_utils import *

''' This script replaces the notebook "prepare_dataset_example.ipynb"
'''

if __name__ == '__main__':
    range_notes = ['C3', 'A#5'] # alto sax range is ['Db3', 'A5'], take half-step below/above
    alto_sax_range = librosa.note_to_hz(range_notes)
    TEST_MODE = False #--- mode for experimenting on a small dataset
    tgt_sr = 44100
    
    if not TEST_MODE:
        data_folder = '/home/mlspeech/itamark/ssynth/git_repos/DeepLearningExamples/PyTorch/SpeechSynthesis/HiFiGAN/data_ssynth/wavs_raw' #'/home/itamar/ssynth/data/wavs'
        flist = glob.glob(f'{data_folder}/*Free*dynamic_mic*.wav')
    else:
        data_folder = '/home/mlspeech/itamark/ssynth/git_repos/DeepLearningExamples/PyTorch/SpeechSynthesis/HiFiGAN/data_ssynth_TMP/wavs_raw' #'/home/itamar/ssynth/data/wavs'
        flist = glob.glob(f'{data_folder}/*dynamic_mic*.wav')
        
    out_dir = data_folder.replace('wavs_raw', 'wavs')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    print_info = True
    dur = 0
    if print_info:
        print(f'found {len(flist)} files in wavs_raw folder:')
        print(pd.Series([os.path.basename(fl) for fl in flist]))
        print('\n')
        for fnm in flist:
            f = sf.SoundFile(fnm)
            sec = f.frames / f.samplerate
            dur += sec
            #print(f'samples = {f.frames}')
            print(f'file {os.path.basename(fnm)}')
            print(f'\tsample rate = {f.samplerate}, sample format = {f.subtype_info}, seconds = {sec:.1f}')
    print(f'total recording duration {dur / 60:.1f} minutes')
    print(f'NOTE: output files will be saved to {out_dir}')
    
    save_to_disk = True #if TEST_MODE else False
    process_if_exists = True # False
    
    #--- WARNING this will overrite existing raw audio files!
    write_resampled_files_to_disk = False # True 
    
    phrase_df = []
    
    for file_nm in flist:
        #--- check for previous results
        file_nm_base = os.path.basename(file_nm)
        exist_wavs = glob.glob(f'{out_dir}/{file_nm_base[:-4]}*.wav')
        
        if not process_if_exists and len(exist_wavs) > 0:
            print(f'found {len(exist_wavs)} phrases from file {file_nm_base}. Skipping.')
            continue
      
        print(f'>>> loading {file_nm_base}')
        f = sf.SoundFile(file_nm)
        resampling = False
        if f.samplerate != tgt_sr:
        resampling = True
        print(f'file sr is {f.samplerate}, resampling to {tgt_sr}')
    
    y, sr = librosa.load(file_nm, sr = tgt_sr) #--- NOTE librosa converts 24 bit audio (which is int32) to float in [-1, 1]
    if resampling and write_resampled_files_to_disk:
        print('Warning: overwriting existing file with resampled one')
        with sf.SoundFile(file_nm, 'w', tgt_sr, 1, f.subtype) as fout:
            fout.write(y)
    
    phrase_inds, seg_dur_sec = split_audio_to_phrases(y, sr)
    
    print(f'Total phrases: {len(phrase_inds)}')
    print(f'phrase durations sec: min {seg_dur_sec.min():.1f} max {seg_dur_sec.max():.1f} mean {seg_dur_sec.mean():.1f}')

    #--- save phrases to disk
    
    if save_to_disk:
        print(f'saving phrases to {out_dir}')
    for k, pind in enumerate(phrase_inds):
        yout = y[pind[0]:pind[1]]
        ifnm = file_nm_base.replace('.wav', f'_phrase{k:03d}.wav')
        fnm_out = f'{out_dir}/{ifnm}'
        phrase_df.append(pd.Series(
            dict(file_nm = file_nm_base, 
                 phrase_id = ifnm.replace('.wav', ''), 
                 sample_start = pind[0], 
                 sample_end = pind[1])))
        if save_to_disk:
            sf.write(fnm_out, yout, sr, subtype = f.subtype)
            
print('done')

#--- write phrase dataframe to disk
if len(phrase_df) > 0:
    phrase_df = pd.concat(phrase_df, axis = 1).T
    phrase_df.sort_values(by = 'phrase_id', inplace = True)
all_files = glob.glob(f'{out_dir}/*.wav')
print(f'data set size: {len(all_files)} phrases')

phrase_df_fnm = f'{data_folder}/../phrase_df.csv'

if not os.path.isfile(phrase_df_fnm): # False:
    phrase_df.to_csv(phrase_df_fnm)
    print(f'saved phrase dataframe to {phrase_df_fnm}')
else:
    phrase_df = pd.read_csv(phrase_df_fnm)

#display(phrase_df.head())

#--- metadata.csv used by HiFiGan training script
metadata_fnm = f'{data_folder}/../metadata.csv'
if not os.path.isfile(metadata_fnm):
    print(f'writing metadata.csv to {metadata_fnm}')
    (phrase_df['phrase_id'] + '||').to_csv(metadata_fnm, index=False, header=False)
else:
    print(f'file {metadata_fnm} exists, not writing a new one')

if False:
    #--- write it below, when we have midi data as well
    #--- filelist used by FastPitch (in HiFiGan there's a script that create file lists)
    filelist_fnm = f'{data_folder}/../filelists_fastpitch/ssynth_audio.txt'
    if not os.path.isfile(filelist_fnm):
        ('wavs/' + phrase_df['phrase_id'] + '.wav|').to_csv(filelist_fnm, index=False, header=False)
    else:
        print(f'file {filelist_fnm} exists, not writing a new one')

#--- validation - compare detected phrses from phrase_df to actual files on disk
for file_nm in flist:
    file_nm_base = os.path.basename(file_nm)
    print(f'>>>> file {file_nm_base}')
    y, sr = librosa.load(file_nm, sr = tgt_sr) 
    pdf = phrase_df.query("file_nm == @file_nm_base")
    print(f'total file duration {y.shape[0] / tgt_sr / 60:.1f} min')
    print(f'total phrase duration {(pdf.sample_end - pdf.sample_start).sum()/tgt_sr/60:.1f} min')
    print(f'phrases: {pdf.shape[0]}')

    tlen = 0
    for k in range(pdf.shape[0]):
        p = pdf.iloc[k]
        p_fnm = f'{out_dir}/{p.phrase_id}.wav'
        y_p, _ = librosa.load(p_fnm, sr = tgt_sr) 
        tlen += len(y_p)
        if p.sample_end - p.sample_start != len(y_p):
            #pass
            print(f'[{k}] mismatch between detected phrase and pharse-file on disk')

        if False and k==24:
            print(f'phrase from {os.path.basename(p_fnm)} ({len(y_p)} samples)')
            ipd.display(ipd.Audio(y_p, rate=sr))
            print(f'phrase from detection ({p.sample_end-p.sample_start} samples)')
            ipd.display(ipd.Audio(y[p.sample_start:p.sample_end], rate=sr))
            break
    print(f'done, total len {tlen/tgt_sr/60:.1f} min')

import soundfile as sf
import pandas as pd
import glob
import os

import librosa.display

from phrase_utils import *
from ssynth.utils import midi

''' This script replaces the notebook "prepare_dataset_example.ipynb"
    Take list of (long) recordings and split to phrases, save pharses and info to disk
'''

if __name__ == '__main__':
    range_notes = midi.MidiUtils.alto_sax_range_pmhs_notes #['C3', 'A#5'] # alto sax range is ['Db3', 'A5'], take half-step below/above
    alto_sax_range = librosa.note_to_hz(range_notes)
    TEST_MODE = False #--- mode for experimenting on a small dataset
    tgt_sr = 44100
    validate_files = True
    
    if not TEST_MODE:
        data_folder = '/home/mlspeech/itamark/ssynth/git_repos/DeepLearningExamples/PyTorch/SpeechSynthesis/HiFiGAN/data_ssynth/wavs_raw' #'/home/itamar/ssynth/data/wavs'
        flist = glob.glob(f'{data_folder}/*Free*dynamic_mic*.wav')
    else:
        data_folder = '/home/mlspeech/itamark/ssynth/git_repos/DeepLearningExamples/PyTorch/SpeechSynthesis/HiFiGAN/data_ssynth_TMP/wavs_raw' #'/home/itamar/ssynth/data/wavs'
        flist = glob.glob(f'{data_folder}/*dynamic_mic*.wav')
    
    flist.sort()
        
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
    process_if_exists = False #True # False
    
    #--- WARNING this will overrite existing raw audio files!
    write_resampled_files_to_disk = False # True 
    
    phrase_df = pd.DataFrame(columns = ['file_nm', 'phrase_id', 'sample_start', 'sample_end'])
    #--- read existing pharse data
    phrase_df_fnm = f'{data_folder}/../phrase_df.csv'
    phrase_df_existing = pd.read_csv(phrase_df_fnm, index_col = 0)
    print(f'found {phrase_df_existing.shape[0]} phrases from current files {phrase_df_existing.file_nm.unique()}')
    
    for file_nm in flist:
        #--- check for previous results
        file_nm_base = os.path.basename(file_nm)
        exist_wavs = glob.glob(f'{out_dir}/{file_nm_base[:-4]}*.wav')
        n_found = len(exist_wavs)
        
        if not process_if_exists and n_found > 0:
            phrase_df_curr = phrase_df_existing.query('file_nm == @file_nm_base') 
            assert phrase_df_curr.shape[0] == n_found, 'number of phrase files on disk and in data frame must be the same'
            print(f'found {n_found} phrases from file {file_nm_base}, on disk and in phrase_df. Skipping file.')
            phrase_df = pd.concat([phrase_df, phrase_df_curr], axis = 0, ignore_index = True) 
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
            new_row = pd.DataFrame(
                dict(file_nm = file_nm_base, 
                     phrase_id = ifnm.replace('.wav', ''), 
                     sample_start = pind[0], 
                     sample_end = pind[1]),
                index = [0])
            phrase_df = pd.concat([phrase_df, new_row], ignore_index = True)
            if save_to_disk:
                sf.write(fnm_out, yout, sr, subtype = f.subtype)
                
    print('done')

    #--- write phrase dataframe to disk
    if len(phrase_df) > 0:
        phrase_df.sort_values(by = 'phrase_id', inplace = True)
        
    all_files = glob.glob(f'{out_dir}/*.wav')
    print(f'phrase_df size: {phrase_df.shape[0]}, phrase files on disk: {len(all_files)}')

    phrase_df_fnm = f'{data_folder}/../phrase_df.csv'
    metadata_fnm = f'{data_folder}/../metadata.csv'
    #if not os.path.isfile(phrase_df_fnm): # False:
    if phrase_df.shape[0] > phrase_df_existing.shape[0]:
        print(f'saving new phrase data-frame to {phrase_df_fnm}')
        phrase_df.to_csv(phrase_df_fnm)
        #--- metadata.csv is used by HiFiGan training script
        print(f'writing new metadata to {metadata_fnm}')
        (phrase_df['phrase_id'] + '||').to_csv(metadata_fnm, index=False, header=False)
    else:
        print('Did not found new phrases, not writing "phrase_df" data-frame to disk')
        #phrase_df = pd.read_csv(phrase_df_fnm)

    #--- we don't use Fastpitch 
    #--- filelist used by FastPitch (in HiFiGan there's a script that create file lists)
    #filelist_fnm = f'{data_folder}/../filelists_fastpitch/ssynth_audio.txt'
    #if not os.path.isfile(filelist_fnm):
    #    ('wavs/' + phrase_df['phrase_id'] + '.wav|').to_csv(filelist_fnm, index=False, header=False)
    #else:
    #    print(f'file {filelist_fnm} exists, not writing a new one')

    #--- validation - compare detected phrses from phrase_df to actual files on disk
    if validate_files:
        print('Validating files on disk vs. phrase data-frame')
        tlen = 0
        for file_nm in flist:
            file_nm_base = os.path.basename(file_nm)
            print(f'>>>> file {file_nm_base}')
            y, sr = librosa.load(file_nm, sr = tgt_sr) 
            pdf = phrase_df.query("file_nm == @file_nm_base")
            print(f'file duration {y.shape[0] / tgt_sr / 60:.1f} min, phrase duration {(pdf.sample_end - pdf.sample_start).sum() / tgt_sr / 60.:.1f} min, num phrases {pdf.shape[0]}')

            for k in range(pdf.shape[0]):
                p = pdf.iloc[k]
                p_fnm = f'{out_dir}/{p.phrase_id}.wav'
                y_p, _ = librosa.load(p_fnm, sr = tgt_sr) 
                tlen += len(y_p)
                if p.sample_end - p.sample_start != len(y_p):
                    #pass
                    print(f'[{k}] mismatch between detected phrase and pharse-file on disk')

        print(f'done, total len {tlen / tgt_sr / 60.:.1f} min')

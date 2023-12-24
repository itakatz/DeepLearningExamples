import numpy as np
     
if __name__ == '__main__':
    #--- exactly one of these should be None:
    num_harmonics = None # 10
    max_freq_hz = 16000
    spline_smoothing = 0.5 # set to None for no smoothing of amplitude envelope
    
    suffix = f'{num_harmonics}h' if max_freq_hz is None else f'{int(max_freq_hz / 1000)}k'
    suffix += '' if spline_smoothing is None else f'_spl{spline_smoothing}'
    
    synth_out_dir = data_folder.replace('wavs_raw', f'wavs_synth_{suffix}')
    pitch_out_dir = data_folder.replace('wavs_raw', 'pitch_synth')
    print(f'writing synthesized wavs to {synth_out_dir}')
    print(f'writing extracted pitch to {pitch_out_dir}')
    
    if not os.path.isdir(synth_out_dir):
        os.mkdir(synth_out_dir)
    if not os.path.isdir(pitch_out_dir):
        os.mkdir(pitch_out_dir)
    
    #--- iterate over files, and over phrases in an inner loop 
    for ifnm, fnm in enumerate(flist):
        #if ifnm < 2:
        #    continue
        fnm_base = os.path.basename(fnm)
        midi_fnm = fnm.replace('/wavs_raw/', f'/{midi_folder}/').replace('.wav', '.mid')
        if TEST_MODE and '_dynamic_mic' in midi_fnm:
            midi_fnm = midi_fnm.replace('_dynamic_mic', '')
            
        print(f'[{ifnm}] reading midi file {os.path.basename(midi_fnm)}')
        midi_df = read_midi_to_df(midi_fnm)
        verify_midi(midi_df)
        p_df = phrase_df.query("file_nm == @fnm_base").reset_index(drop = True)
        print(f'processing {p_df.shape[0]} phrases')
        for iphrs, phrs in p_df.iterrows():             
            #if iphrs < 610:
            #    continue
            wav_fnm = f'{out_dir}/{phrs.phrase_id}.wav'
            seg, sr = librosa.load(wav_fnm, sr = tgt_sr)
            midi_p = midi_phrase_from_dataframe(phrs, midi_df, sr)
            t0 = phrs.sample_start / sr
            try:
                seg_synth, env, pitch = phrase_to_synth(seg, sr, midi_p, t0, 
                                                        num_harmonics = num_harmonics, 
                                                        max_freq_hz = max_freq_hz,
                                                        spline_smoothing = spline_smoothing,  
                                                        verbose = False)
            except Exception as e:
                print(f'phrase {iphrs} failed with error: {e}')
                continue
            #--- save synth signal and pitch
            fnm_out = f'{synth_out_dir}/{phrs.phrase_id}.wav'
            sf.write(fnm_out, seg_synth, sr, subtype = 'PCM_24')
            pitch_fnm_out = f'{pitch_out_dir}/{phrs.phrase_id}.pt'
            pitch = torch.tensor(pitch[np.newaxis,:].astype(np.float32))
            torch.save(pitch, pitch_fnm_out)
            #break
        #break
import numpy as np
import pandas as pd
import librosa
import mido

class MidiUtils:
    alto_sax_range_notes = ['Db3', 'A5'] # alto sax range is ['Db3', 'A5']
    alto_sax_range_hz = librosa.note_to_hz(alto_sax_range_notes)
    #--- helper range of plus/minus half step above/below (aka "pmhs")
    alto_sax_range_pmhs_notes = ['C3', 'A#5']
    alto_sax_range_pmhs_hz = librosa.note_to_hz(alto_sax_range_pmhs_notes)

def binary_array_to_seg_inds(arr, shift_end_ind = True):
    seg_inds = np.diff(np.r_[0, np.int_(arr), 0]).nonzero()[0]
    n_segs = int(seg_inds.shape[0] / 2)
    seg_inds = seg_inds.reshape((n_segs, 2)) # + np.c_[np.zeros(n_segs),-np.ones(n_segs)]   
    if shift_end_ind:
        seg_inds[:,1] -= 1
    return seg_inds    

def read_midi_to_df(midi_fnm, try_to_fix_note_order = True, time_offset_sec = 0.):
    mid = mido.MidiFile(midi_fnm)
    
    #assert(len(mid.tracks) == 1)
    tr = mido.merge_tracks(mid.tracks)
    df =  pd.DataFrame([m.dict() for m in tr])
    tempo = df.set_index('type').loc['set_tempo','tempo']
    if type(tempo) == pd.Series:
        uniq_tempo = tempo.unique()
        if len(uniq_tempo) > 1:
            raise Exception('multiple tempo changes not supported')
        else:
            tempo = uniq_tempo[0]
            
    df['ts_sec'] = mido.tick2second(df.time.cumsum(), mid.ticks_per_beat, tempo)
    if time_offset_sec != 0.:
        df['ts_sec'] += time_offset_sec
        df = df[df['ts_sec'] >= 0.]
        
    #--- extract controls (pitch and aftertouch has seperated messages, not included in CC)
    df_aftertouch = df[df.type == 'aftertouch'].dropna(axis = 1).reset_index(drop = True)
    df_pitch = df[df.type == 'pitchwheel'].dropna(axis = 1).reset_index(drop = True)
    df_cc = df[df.type == 'control_change'].dropna(axis = 1).reset_index(drop = True)
    
    #--- some mete-messages like "channel prefix" contain non-zero time value. so remove them *after* calculating 'ts_sec'
    for type_remove in ['channel_prefix', 'track_name', 'instrument_name', 'time_signature', 'key_signature', 
                        'smpte_offset', 'set_tempo', 'end_of_track', 'midi_port', 'program_change', 'control_change', 'pitchwheel', 'aftertouch', 'marker']:
        df = df[df.type != type_remove]
    
    df = df.dropna(axis = 1).reset_index(drop = True)
    
    #--- sometimes, instead of a sequence of on-off notes, we get on-on-off-off. try to fix that
    if try_to_fix_note_order:
        try:
            verify_midi(df)
        except AssertionError:
            print(f'{midi_fnm}: note order problem in midi dataframe, trying to fix...')
            df_copy = df.copy()
            for ii, inote in df.iterrows():
                if inote.type == 'note_on':
                    assoc_note_off = df.iloc[ii:].query('type == "note_off" and note == @inote.note')
                    if len(assoc_note_off) == 0:
                        raise Exception('note on with no associated note off')
                    inote_off = assoc_note_off.iloc[0]
                    inote_off_ind = inote_off.name
                    if inote_off_ind > ii + 1:
                        next_note_on = df.iloc[ii+1:].query('type == "note_on"')
                        if len(next_note_on) > 0:
                            next_note_on = next_note_on.iloc[0]
                            if next_note_on.ts_sec < inote_off.ts_sec:
                                df.loc[inote_off_ind, 'ts_sec'] = next_note_on.ts_sec - .001        
#             #--- indices of where we expect to see "note off" and see "note on"
#             off_err_ind = df[((df.index % 2) == 1) & (df.type == 'note_on')].index
#             for ind in off_err_ind:
#                 curr_note = df.loc[ind]
#                 next_note = df.loc[ind + 1]
#                 prev_note = df.loc[ind - 1]
#                 if next_note.type == 'note_off' and next_note.note == prev_note.note:
#                     df.loc[ind + 1, 'ts_sec'] = curr_note.ts_sec - 0.001
            df = df.sort_values(by = 'ts_sec', kind = 'stable').reset_index(drop = True)
            try:
                verify_midi(df)
                print('fixed')
            except AssertionError:
                print('fix failed, calling verify_midi() on returned dataframe will fail')
                #--- if fix failed, return the original copy
                df = df_copy
                
    return df, df_pitch, df_aftertouch, df_cc

def verify_midi(midi_df):
    #--- validate the assumption that we have series of note-on/note-off events
    assert((midi_df['type'].iloc[::2] == 'note_on').all() and 
       (midi_df['type'].iloc[1::2] == 'note_off').all() and
       (midi_df['note'].iloc[::2].to_numpy() == midi_df['note'].iloc[1::2].to_numpy()).all())

def midi_phrase_from_dataframe(p, midi_df, sr):
    t0 = p.sample_start / sr
    t1 = p.sample_end / sr
    midi_p = midi_df[(midi_df.ts_sec >= t0) & (midi_df.ts_sec <= t1)]
    if midi_p.shape[0] == 0:
        return midi_p
    
    #--- check for missing note_off (at end) or note_on (at start)
    first_note = midi_p.iloc[0]
    if first_note['type'] == 'note_off':
        candidate = midi_df.loc[first_note.name - 1]
        if candidate['type'] == 'note_on' and candidate['note'] == first_note['note']:
            midi_p = pd.concat([candidate.to_frame().T, midi_p])
            
    last_note = midi_p.iloc[-1]
    if last_note['type'] == 'note_on':
        candidate = midi_df.loc[last_note.name + 1]
        if candidate['type'] == 'note_off' and candidate['note'] == last_note['note']:
            midi_p = pd.concat([midi_p, candidate.to_frame().T])
    
    return midi_p
    
def phrase_to_midi_string(p, midi_df, sr):    
    midi_p = midi_phrase_from_dataframe(p, midi_df, sr)            
    try:
        verify_midi(midi_p)
    except Exception as e:
        print(f'phrase {p.phrase_id} verification failed')
        return ''
    
    note_on = midi_p.loc[midi_p.type == 'note_on']
    s = f"wavs/{p.phrase_id}.wav|{' '.join(note_on.note.astype(int).astype(str).to_list())}"
    return s
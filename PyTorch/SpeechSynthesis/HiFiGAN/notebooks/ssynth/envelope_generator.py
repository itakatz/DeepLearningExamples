import set_python_path
from utils.synthesis import wav_midi_to_synth
from utils import midi

import os
import pandas as pd
import numpy as np
import librosa

if __name__ == '__main__':
    data_dir = '../../data_ssynth'

    file_id = '01_Free_Improv_dynamic_mic'
    midi_fnm = f'{data_dir}/auto_midi/{file_id}.mid'
    phrase_df_fnm = f'{data_dir}/phrase_df.csv'
    print(f'reading midi file {os.path.basename(midi_fnm)}')
    midi_df, midi_pitch, midi_aftertouch, midi_cc = midi.read_midi_to_df(midi_fnm)
    midi.verify_midi(midi_df)

    phrase_df = pd.read_csv(phrase_df_fnm, index_col = 0).reset_index(drop = True)
    phrase_df = phrase_df[phrase_df.file_nm.str.contains(file_id)]

    phrase_ind = 14 #26
    p = phrase_df.iloc[phrase_ind]

    wav_fnm = f'{data_dir}/wavs/{p.phrase_id}.wav'
    #seg, sr = librosa.load(wav_fnm, sr = sampling_rate)
    seg, sampling_rate = librosa.load(wav_fnm, sr = None)

    t0 = p.sample_start / sampling_rate
    midi_p = midi.midi_phrase_from_dataframe(p, midi_df, sampling_rate)

    #--- get envelope 
    max_freq_hz = 16000
    pd_cfg = dict(win = 1024, 
                  ac_win = 512, # autocorrelation window
                  hop = 256)

    _x, _env, freq, raw_pitch = wav_midi_to_synth(seg, sampling_rate, midi_p, t0, pd_cfg, max_freq_hz, spline_smoothing = 2, verbose = False)
    env = librosa.feature.rms(y = seg, frame_length = 512, hop_length = 256, center = True)
    env = 1.3 * np.sqrt(2) * env[0] 
    
    #--- make plot (to view, run command "python -m http.server 8080 &> /dev/null &", forward the 8080 port from remote to local, and open the browser at localhost:8080)
    import plotly.subplots as subplots
    import plotly.graph_objects as go
    from IPython.display import Audio
    t_env = librosa.frames_to_time(np.arange(env.shape[0]), sr = sampling_rate,hop_length = 256, n_fft = 512)
    t_seg = librosa.samples_to_time(np.arange(seg.shape[0]),sr=sampling_rate)
    fig = subplots.make_subplots(specs=[[{"secondary_y": True}]])
    fig = fig.update_layout(dict(width = 1400, height = 800))
    down_factor = 10
    fig = fig.add_trace(go.Scatter(x = t_seg[::down_factor], y = seg[::down_factor], mode = 'lines', name = 'signal'), secondary_y = False)
    fig = fig.add_trace(go.Scatter(x = t_env, y = env, mode = 'markers', name = 'amplitude env'), secondary_y = False)
    fig = fig.add_trace(go.Scatter(x = t_env, y = raw_pitch['freq'], mode = 'markers', name = 'pitch [Hz]', hovertext = raw_pitch['vprob']), secondary_y = True)
    #cols = ['red','read']
    for i in range(0, midi_p.shape[0], 2):
        row = midi_p.iloc[i]
        col = 'red' #cols[(i // 2) % 2]
        ann = dict(text = f'idx {i // 2}', hovertext = f'note: {int(row.note)}, velocity: {int(row.velocity)}')
        tstart, tend = midi_p.iloc[i].ts_sec - t0, midi_p.iloc[i+1].ts_sec - t0

        fig = fig.add_vrect(x0 = tstart, x1 = tend, fillcolor = col, opacity = 0.2, line_width = 2, annotation = ann, secondary_y = False)
        fig = fig.add_shape(type = 'line', x0 = tstart, x1 = tend, y0 = librosa.midi_to_hz(row.note), y1 = librosa.midi_to_hz(row.note), secondary_y = True, yref = 'y2')

    h1 = fig.to_html(include_plotlyjs = 'cdn', full_html = False)
    h2 = Audio(seg, rate = sampling_rate, normalize = False)._repr_html_()
    with open('../plots/seg_env_audio.html', 'w') as f:
        f.write('\n'.join([h1, h2]))

    #TODO
    # apply 1 simple onset fix: roll onset backwards while:
    # * pitch is within range around note's pitch, [p-dp, p+dp] for some th dp (in semitones!) - NOTE SURE, what about glides? maybe add prob(pitch)?
    # * energy decreases from t to t-1 with a factor of at least de (aka, put a threshold on diff(env))
    # no note offset is reached


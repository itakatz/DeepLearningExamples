import numpy as np
import librosa
from scipy.signal import decimate, butter, dlti # resample_poly
from scipy.interpolate import UnivariateSpline, interp1d

from .midi import MidiUtils, binary_array_to_seg_inds

''' This module contains methods to synthesize various signals that can be used as input to a HiFiGAN model
    Generally, methods which generate final audio using a trained hifigan model do *not* belong here.
'''

def get_num_harmonics(min_freq_src_hz, max_freq_src_hz, sr, max_freq_tgt_hz):
    fmin = max(MidiUtils.alto_sax_range_hz[0], min_freq_src_hz)
    
    num_harmonics = int(max_freq_tgt_hz / fmin)
    new_sr = 2 * max_freq_src_hz * num_harmonics
    #--- take the smallest multiple of sr which is high enough (6 is the highest, assuming freqs.max() <= 932 Hz)
    new_sr_factor = [k for k in range(1, 10) if k * sr > new_sr][0]
    return num_harmonics, new_sr_factor

def additive_synth_sawtooth(freq, env, sampling_rate, additive_synth_k = None, max_freq_hz = None):
    ''' TDOO add code to synthesize up to f_max (and not a given number of harmonics)
        given input frequency and envelope sampled at sampling_rate, synthesize a band-limited
        sawtooth wave using additive synthesis of 10 (or k) harmonies
    '''    
    #--- set number of harmonics of sawtooth wave
    if additive_synth_k is not None:
        should_downsample = False
        sampling_rate_new = None
    else:
        num_harmonics, new_sr_factor = get_num_harmonics(freq[freq > 20].min(), freq.max(), sampling_rate, max_freq_hz)
        #--- make sure we stay below new nyquist
        assert freq.max() * num_harmonics < 0.5 * sampling_rate * new_sr_factor, f'Nyquist says you cannot synthesize {num_harmonics} harmonics at {new_sr_factor} X (current sampling rate)'
        additive_synth_k = num_harmonics
        sampling_rate_new = sampling_rate * new_sr_factor
        should_downsample = True
    
    dt = 1 / sampling_rate
    
    #--- interpolate (upsample) to sampling-rate grid, if needed
    if sampling_rate_new is not None:
        tmax = len(freq) * dt
        t_old = np.arange(0, len(freq)) * dt #np.arange(0, tmax, dt)
        fintrp = interp1d(t_old, freq)
        dt = 1 / sampling_rate_new
        t_new = np.arange(0, tmax, dt)
        t_new = t_new[(t_new <= t_old.max()) & (t_new >= t_old.min())] # avoid interpolation out of bounds
        freq = fintrp(t_new)  

    #--- phase is the integral of instantanous freq
    phi = np.cumsum(2 * np.pi * freq * dt)
    # to wrap: phi = (phi + np.pi) % (2 * np.pi) - np.pi 
        
    x = np.sin(phi) #(np.sin(phi) + .5*np.sin(2*phi) + .333*np.sin(3*phi) + .25*np.sin(4*phi))
    for k in range(2, additive_synth_k + 1):
        x += (-1)**(k-1) * np.sin(k * phi) / k
    
    #--- if we upsampled, go back to original rate
    if should_downsample:
        #--- for x, give a "anti-alias" filter to "decimate", but actually use it to filter above the desired max_freq_hz
        zpk = butter(12, max_freq_hz, output = 'zpk', fs = sampling_rate_new)
        aa_filt = dlti(*zpk) 
        x = decimate(x, new_sr_factor, ftype = aa_filt)
        freq = decimate(freq, new_sr_factor) #--- fnew is just used to zero the envelope, so decimate so size fits
        #sr = int(sr / new_sr_factor)
    
    x *= env
    
    return x

def wav_midi_to_synth(seg, sr, midi_p, t0, pitch_detection_cfg, num_harmonics = None, max_freq_hz = None, spline_smoothing = None, verbose = False):
    ''' This method takes raw wav (an audio segment, aka "seg") and an associated midi phrase midi_p as input, and returns a synthesized wave with same amplitude and frequency envelopes
        The midi phrase is used to enclose notes inside a [note_on, note_off] interval
        (in the original notebook this methid is called "seg_to_synth")
    
        Exactly **one** of these 2 should be given (and the other set to None):
            - num harmonics: how many harmonics (inc the fundamental) are used in the saw-tooth additive synthesis
                             in this case the max-freq is note-dependent (f0*num_harmonics) and the caller is responsible
                             to make sure that (highest note in hz) * (num_harmonics) < nyquist
            - max_freq_hz:   synthesize up to this frequency. This is done by upsampling, synthesizing the required amound of harmonics,
                             and downsampling back to sr
        Other arguments:                    
            - smooth_env:    flag to apply smooting using 2nd order splines
    '''
    
    assert(num_harmonics is None or max_freq_hz is None)
    
    win = pitch_detection_cfg['win']
    ac_win = pitch_detection_cfg['ac_win']
    hop = pitch_detection_cfg['hop']    
    range_hz = MidiUtils.alto_sax_range_pmhs_hz
    
    if verbose:
        print(f'pitch detection range: {range_hz.round(1)} Hz, {(sr / range_hz).astype(int)} samples')
        print(f'pitch detection: frame len {win}, auto-corr len {ac_win} (min freq of {sr / ac_win:.1f} Hz), hop len {hop}')
            
    f1, vflag1, vprob1 = librosa.pyin(seg, 
                                      fmin = range_hz[0], 
                                      fmax = range_hz[1], 
                                      sr = sr, 
                                      frame_length = win, 
                                      win_length = ac_win, 
                                      hop_length = hop, 
                                      center = True, 
                                      max_transition_rate = 100)
    times1 = librosa.times_like(f1, sr = sr, hop_length = hop)
    no_note1 = (~vflag1)
    tmin = times1[0]
    tmax = times1[-1]
    
    note_on = midi_p.loc[midi_p.type == 'note_on']
    note_off = midi_p.loc[midi_p.type == 'note_off']
    #note_hz = librosa.midi_to_hz(note_on.note)
    note_on_ts = note_on['ts_sec'].values - t0
    note_off_ts = note_off['ts_sec'].values - t0
    
    #-------------------------------------------------------------------------------------------------------------------
    #--- interpolate missing pitch, where possible. otherwise, set to 0 (in order to accumulate 0 phase when integrating)
    #-------------------------------------------------------------------------------------------------------------------
    #--- step A, interpolate within (intra-) midi notes
    n_notes = note_on.shape[0]
    if verbose:
        print(f'samples with non-detected pitch: {np.isnan(f1).sum()}')
    for k in range(n_notes):
        #--- first, find missing pitch samples which are inside a detected midi note
        midi_note_span = (times1 >= note_on_ts[k]) & (times1 <= note_off_ts[k])
        
        #--- if no missing pitch samples are in the midi note span, we don't need this note, so skip
        if not (midi_note_span & no_note1).any():
            continue
        
        #--- if we don't have at least 2 pitch samples in the note span, we can't extrapolate, so skip
        if (midi_note_span & ~no_note1).sum() < 2:
            continue
            
        #--- build the interpolating function from detected pitch samples
        pitch_intrp = interp1d(times1[midi_note_span & ~no_note1], 
                               f1[midi_note_span & ~no_note1], 
                               fill_value = 'extrapolate', 
                               kind = 'nearest',
                               assume_sorted = True)
        #--- the time samples where we want to interpolate: inside midi note AND missing pitch
        t_intrp = times1[midi_note_span & no_note1]
        f1[midi_note_span & no_note1] = pitch_intrp(t_intrp)

    if verbose:
        print(f'after interpolating using midi notes: samples with non-detected pitch: {np.isnan(f1).sum()}')

    #--- step B, interpolate across (inter-) midi notes
    max_gap_to_interpolate_sec = 0.1 #--- don't interpolate gaps above this interval in seconds
    no_note1 = np.isnan(f1)
    seg_inds = binary_array_to_seg_inds(no_note1, shift_end_ind = False)
    seg_lens_sec = np.diff(seg_inds, 1)[:,0] * hop / sr
    for k, inds in enumerate(seg_inds):
        #--- don't interpolate head or tail of signal, or if gap is too long
        #--- TODO check energy envelope in gap (interpolate only above env threshold)
        gap_len = seg_lens_sec[k]
        if (inds[0] == 0) or (inds[1] == len(f1)) or gap_len > max_gap_to_interpolate_sec:
            continue
        gap_len_samples = inds[1] - inds[0]
        if verbose:
            print(f'interpolating over {gap_len_samples} samples over gap of {gap_len:.3f} sec')
        #--- linear interpolation using 1 sample before and after
        new_freqs = np.linspace(f1[inds[0] - 1], f1[inds[1]], gap_len_samples + 2)
        f1[inds[0]:inds[1]] = new_freqs[1:-1]

    no_note1 = np.isnan(f1)
    seg_inds = binary_array_to_seg_inds(no_note1, shift_end_ind = False)
    if verbose:
        print(f'after interpolating over small gaps: samples with non-detected pitch: {np.isnan(f1).sum()}')
    #--- lastly, fill with zeros the samples that are still missing
    f1[np.isnan(f1)] = 0.
    
    #--- set number of harmonics of sawtooth wave
    if num_harmonics is not None:
        additive_synth_k = num_harmonics # 10
        should_downsample = False
    else:
        num_harmonics, new_sr_factor = get_num_harmonics(f1[f1 > 20].min(), f1.max(), sr, max_freq_hz)
        #--- make sure we stay below new nyquist
        assert f1.max() * num_harmonics < 0.5 * sr * new_sr_factor, f'Nyquist says you cannot synthesize {num_harmonics} harmonics at {new_sr_factor} X (current sampling rate)'
        additive_synth_k = num_harmonics
        sr *= new_sr_factor
        should_downsample = True
    
    #--- now interpolate to sampling-rate grid
    dt = 1 / sr
    fintrp = interp1d(times1, f1)
    tnew = np.arange(tmin, tmax, dt)
    fnew = fintrp(tnew)
    
    #--- phase is the integral of instantanous freq
    phi = np.cumsum(2 * np.pi * fnew * dt)
    # to wrap: phi = (phi + np.pi) % (2 * np.pi) - np.pi 
        
    x = np.sin(phi) #(np.sin(phi) + .5*np.sin(2*phi) + .333*np.sin(3*phi) + .25*np.sin(4*phi))
    for k in range(2, additive_synth_k + 1):
        x += (-1)**(k-1) * np.sin(k*phi) / k
    
    #--- if we upsampled, go back to original rate
    if should_downsample:
        #--- for x, give a "anti-alias" filter to "decimate", but actually use it to filter above the desired max_freq_hz
        zpk = butter(12, max_freq_hz, output = 'zpk', fs = sr)
        aa_filt = dlti(*zpk) 
        x = decimate(x, new_sr_factor, ftype = aa_filt)
        fnew = decimate(fnew, new_sr_factor) #--- fnew is just used to zero the envelope, so decimate so size fits
        sr = int(sr / new_sr_factor)
    
    env = librosa.feature.rms(y = seg, frame_length = 512, hop_length = 1, center = True)
    env = 1.3 * np.sqrt(2)*env[0, :len(x)]
    env[fnew == 0] = 0. # don't apply envelope where there was no pitch found

    #--- make envelope go to zero smoothly. This also takes care of the non-continous phase at jumps of f1 to 0
    env_segments = binary_array_to_seg_inds(env == 0)
    decay_time_sec = 0.05 #--- 50 msec decay time
    decay_time_samples = int(decay_time_sec * sr)
    for env_seg in env_segments:
        if env_seg[0] == 0:
            continue
        ind_start = max(0, env_seg[0] - decay_time_samples)
        decay_len = env_seg[0] - ind_start
        decay_factor = np.linspace(1, 0, decay_len)
        env[ind_start: env_seg[0]] *= decay_factor  
    
    if spline_smoothing is not None:
        ts = t0 + np.arange(0, len(env)) / sr
        spl = UnivariateSpline(ts, env, s = spline_smoothing, k = 2)
        env = spl(ts)
        env[env < 0.] = 0.
        
    x *= env
    gain = np.sqrt((x**2).mean()) / np.sqrt((seg**2).mean()) 
    x /= gain
    env /= gain

    raw_pitch = dict(freq = f1, vflag = vflag1, vprob = vprob1)
    
    return x, env, fnew, raw_pitch
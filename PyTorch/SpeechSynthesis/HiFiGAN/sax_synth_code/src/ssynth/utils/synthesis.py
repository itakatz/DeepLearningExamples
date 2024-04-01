import numpy as np
import librosa
from scipy.signal import decimate, butter, dlti # resample_poly
from scipy.signal import sosfiltfilt
from scipy.signal import sawtooth as nbl_sawtooth
from scipy.interpolate import UnivariateSpline, interp1d

from .midi import MidiUtils, binary_array_to_seg_inds

''' This module contains methods to synthesize various signals that can be used as input to a HiFiGAN model
    Generally, methods which generate final audio using a trained hifigan model do *not* belong here.
    TODO
    1. Compare the sawtooth impl in "additive_synth_sawtooth" and inside the method "wav_midi_to_synth". We should use only 1 impl
    2. Add aliased sawtooth (that is, without the anti-alias method of additive synthesis)
    3. Add sine wave
'''

def smooth_env_lowpass(env, sr, cfg, clip_below_min = True):
    filt_order = cfg['order'] #2
    filt_cutoff_hz = cfg['cutoff_hz'] #20
    lowpass_sos = butter(filt_order, filt_cutoff_hz, output = 'sos', fs = sr)
    env_min = env.min()
    env = sosfiltfilt(lowpass_sos, env)
    #--- filtering can output negative samples, so clip from below to min before filtering
    if clip_below_min:
        env[env < env_min] = env_min

    return env

def freq_to_phi(freq, sr, normalized = False, phi0 = 0):
    ''' output phase, "normalized" is in [0, 1], not "normalized" is in [0, 2*pi] '''
    c = 1 if normalized else 2 * np.pi
    dt = 1 / sr
    phi = np.cumsum(c * freq * dt)
    phi -= phi[0]
    phi += phi0
    return phi
    
def freq_to_sawtooth(freq, k_harm, sr, phi0 = 0):
    ''' alias-free saw tooth using additive synthesis using k_harm harmonics 
        You can get a sine-wave, by setting k_harm = 1
        You can get a non band-limited (aliased) sawtooth (abbr. nbl_sawtooth), by setting k_harm = np.inf (as if summing infinite number of harmonics)
    '''
    phi = freq_to_phi(freq, sr, phi0 = phi0)
    if np.isinf(k_harm):
        x = nbl_sawtooth(phi + np.pi)  #--- add pi to use the same convention as I use with the bandlimited sawtooth, that is that sawtooth(0) = 0
        return x

    # to wrap: phi = (phi + np.pi) % (2 * np.pi) - np.pi
    x = np.sin(phi) #(np.sin(phi) + .5*np.sin(2*phi) + .333*np.sin(3*phi) + .25*np.sin(4*phi))
    for k in range(2, k_harm + 1):
        x += (-1)**(k-1) * np.sin(k * phi) / k
    x *= 2 / np.pi
    return x #, phi
    
def get_num_harmonics(min_freq_src_hz, max_freq_src_hz, sr, max_freq_tgt_hz):
    fmin = max(MidiUtils.alto_sax_range_hz[0], min_freq_src_hz)
    
    num_harmonics = int(max_freq_tgt_hz / fmin)
    new_sr = 2 * max_freq_src_hz * num_harmonics
    #--- take the smallest multiple of sr which is high enough (6 is the highest, assuming freqs.max() <= 932 Hz)
    new_sr_factor = [k for k in range(1, 10) if k * sr > new_sr][0]
    return num_harmonics, new_sr_factor

def additive_synth_sawtooth(freq, freq_ts, sampling_rate, additive_synth_k = None, max_freq_hz = None):
    ''' TDOO add code to synthesize up to f_max (and not a given number of harmonics)
        given input frequency and envelope sampled at sampling_rate, synthesize a band-limited
        sawtooth wave using additive synthesis of 10 (or k) harmonies
    '''    
    #--- set number of harmonics of sawtooth wave
    if additive_synth_k is not None:
        should_downsample = False
    else:
        num_harmonics, new_sr_factor = get_num_harmonics(freq[freq > 20].min(), freq.max(), sampling_rate, max_freq_hz)
        #--- make sure we stay below new nyquist
        assert freq.max() * num_harmonics < 0.5 * sampling_rate * new_sr_factor, f'Nyquist says you cannot synthesize {num_harmonics} harmonics at {new_sr_factor} X (current sampling rate)'
        additive_synth_k = num_harmonics
        sampling_rate *= new_sr_factor
        should_downsample = True
    
    #--- interpolate (upsample) to sampling-rate grid, if needed
    tmin, tmax = freq_ts[0], freq_ts[-1]
    fintrp = interp1d(freq_ts, freq, kind = 'nearest', assume_sorted = True)
    dt = 1 / sampling_rate
    t_new = np.arange(tmin, tmax, dt)
    #t_new = t_new[(t_new <= freq_ts[-1]) & (t_new >= freq_ts[0])] # avoid interpolation out of bounds
    freq = fintrp(t_new)  
    
    x = freq_to_sawtooth(freq, additive_synth_k, sampling_rate)
    #--- phase is the integral of instantanous freq
    #phi = np.cumsum(2 * np.pi * freq * dt)
    # to wrap: phi = (phi + np.pi) % (2 * np.pi) - np.pi 
    #    
    #x = np.sin(phi) #(np.sin(phi) + .5*np.sin(2*phi) + .333*np.sin(3*phi) + .25*np.sin(4*phi))
    #for k in range(2, additive_synth_k + 1):
    #    x += (-1)**(k-1) * np.sin(k * phi) / k
    
    #--- if we upsampled, go back to original rate
    if should_downsample:
        #--- for x, give a "anti-alias" filter to "decimate", but actually use it to filter above the desired max_freq_hz
        zpk = butter(12, max_freq_hz, output = 'zpk', fs = sampling_rate)
        aa_filt = dlti(*zpk) 
        x = decimate(x, new_sr_factor, ftype = aa_filt)
        freq = decimate(freq, new_sr_factor) #--- fnew is just used to zero the envelope, so decimate so size fits
        sr = int(sampling_rate / new_sr_factor)
    
    return x, freq

def enhance_pitch_using_midi_phrase(midi_p, pitch, vflag, times, t0, hop, sr, verbose = False):
    no_note = (~vflag)
    tmin = times[0]
    tmax = times[-1]
    
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
        print(f'samples with non-detected pitch: {np.isnan(pitch).sum()}')
    for k in range(n_notes):
        #--- first, find missing pitch samples which are inside a detected midi note
        midi_note_span = (times >= note_on_ts[k]) & (times <= note_off_ts[k])
        
        #--- if no missing pitch samples are in the midi note span, we don't need this note, so skip
        if not (midi_note_span & no_note).any():
            continue
        
        #--- if we don't have at least 2 pitch samples in the note span, we can't extrapolate, so skip
        if (midi_note_span & ~no_note).sum() < 2:
            continue
            
        #--- build the interpolating function from detected pitch samples
        pitch_intrp = interp1d(times[midi_note_span & ~no_note], 
                               pitch[midi_note_span & ~no_note], 
                               fill_value = 'extrapolate', 
                               kind = 'nearest',
                               assume_sorted = True)
        #--- the time samples where we want to interpolate: inside midi note AND missing pitch
        t_intrp = times[midi_note_span & no_note]
        pitch[midi_note_span & no_note] = pitch_intrp(t_intrp)

    if verbose:
        print(f'after interpolating using midi notes: samples with non-detected pitch: {np.isnan(pitch).sum()}')

    #--- step B, interpolate across (inter-) midi notes
    max_gap_to_interpolate_sec = 0.1 #--- don't interpolate gaps above this interval in seconds
    no_note = np.isnan(pitch)
    seg_inds = binary_array_to_seg_inds(no_note, shift_end_ind = False)
    seg_lens_sec = np.diff(seg_inds, 1)[:,0] * hop / sr
    for k, inds in enumerate(seg_inds):
        #--- don't interpolate head or tail of signal, or if gap is too long
        #--- TODO check energy envelope in gap (interpolate only above env threshold)
        gap_len = seg_lens_sec[k]
        if (inds[0] == 0) or (inds[1] == len(pitch)) or gap_len > max_gap_to_interpolate_sec:
            continue
        gap_len_samples = inds[1] - inds[0]
        if verbose:
            print(f'interpolating over {gap_len_samples} samples over gap of {gap_len:.3f} sec')
        #--- linear interpolation using 1 sample before and after
        new_freqs = np.linspace(pitch[inds[0] - 1], pitch[inds[1]], gap_len_samples + 2)
        pitch[inds[0]:inds[1]] = new_freqs[1:-1]

    #no_note1 = np.isnan(f1)
    #seg_inds = binary_array_to_seg_inds(no_note1, shift_end_ind = False)
    if verbose:
        print(f'after interpolating over small gaps: samples with non-detected pitch: {np.isnan(pitch).sum()}')
    #--- lastly, fill with zeros the samples that are still missing
    pitch[np.isnan(pitch)] = 0.
    
    return pitch

def wav_midi_to_synth(seg, sr, midi_p, t0, pitch_detection_cfg, num_harmonics = None, max_freq_hz = None, smoothing = None, verbose = False):
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
            - smoothing:    If not None, should be a dict of smoothing config
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
                                      resolution = 0.05, #--- 5 cents pitch resolution
                                      center = True, 
                                      max_transition_rate = 100)
    
    times1 = librosa.times_like(f1, sr = sr, hop_length = hop)
    f1 = enhance_pitch_using_midi_phrase(midi_p, f1, vflag1, times1, t0, hop, sr)
    x, fnew = additive_synth_sawtooth(f1, times1, sr, num_harmonics, max_freq_hz)
    
    #--- that code was moved to additive_synth_sawtooth method:
    #--- set number of harmonics of sawtooth wave
    if False:
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
        fintrp = interp1d(times1, f1, kind = 'nearest', assume_sorted = True)
        tnew = np.arange(tmin, tmax, dt)
        fnew = fintrp(tnew)
        
        x = freq_to_sawtooth(fnew, additive_synth_k, sr)
        
        #--- if we upsampled, go back to original rate
        if should_downsample:
            #--- for x, give a "anti-alias" filter to "decimate", but actually use it to filter above the desired max_freq_hz
            zpk = butter(12, max_freq_hz, output = 'zpk', fs = sr)
            aa_filt = dlti(*zpk) 
            x = decimate(x, new_sr_factor, ftype = aa_filt)
            fnew = decimate(fnew, new_sr_factor) #--- fnew is just used to zero the envelope, so decimate so size fits
            sr = int(sr / new_sr_factor)
   
        assert np.allclose(x,x_), 'x != x_, debug'

    env_hop = 1 # we need the env at audio rate
    env_frame = 512 # this equals ac_win of the pitch detection.. should we set it to ac_win?
    env = librosa.feature.rms(y = seg, frame_length = env_frame, hop_length = env_hop, center = True)
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
    
    if smoothing is not None:
        if smoothing['method'] == 'spline':
            s_param = smoothing['smoothing']
            k_param = smoothing['k']
            ts = t0 + np.arange(0, len(env)) / sr
            spl = UnivariateSpline(ts, env, s = s_param, k = k_param)
            env = spl(ts)
            env[env < 0.] = 0.
        elif smoothing['method'] == 'lowpass':
            env_sr = sr / env_hop 
            env = smooth_env_lowpass(env, env_sr, smoothing)
        else:
            raise ValueError('smoothing cfg shold be a dict with a "method" key')
        
    
    x *= env
    gain = np.sqrt((x**2).mean()) / np.sqrt((seg**2).mean()) 
    x /= gain
    env /= gain

    #--- return envelopes sampled at low rate
    env_out = env[::hop] #--- this is what we would get if we used hop_length > 1 in librosa.feature.rms since I set "center=True"
    n = min(len(f1), len(env_out))
    env_out = env_out[:n]
    f1 = f1[:n]
    pitch_out = dict(freq = f1, vflag = vflag1, vprob = vprob1)
    
    return x, env_out, pitch_out

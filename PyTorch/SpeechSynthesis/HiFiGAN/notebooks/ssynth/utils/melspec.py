from ssynth import set_python_path
from hifigan.data_function import mel_spectrogram

from functools import partial

#--- NOTE starting 2023-05-28, I added a flag to the dataset-creation script, that makes it use the same mel implementation (unless told explicitly not to)
#         for both run-time calculation and save-to-disk calculation

#--- this is the implementation used to generate pre-calculated mels for synthetic wavs/fine-tuning (loaded from disk during training)
class UnknownMelSpecImplementation(Exception): pass
class MelSpecClassNotInitialized(Exception): pass

class MelSpec:
    def __init__(self):
        self.initialized = False
        
    def init(self, cfg, MEL_IMPL):
        filter_length = cfg['filter_length']
        hop_length = cfg['hop_length']
        win_length = cfg['win_length']
        n_mel_channels = cfg['num_mels']
        sampling_rate = cfg['sampling_rate']
        mel_fmin = cfg['mel_fmin']
        mel_fmax = cfg['mel_fmax']

        self.MEL_IMPL = MEL_IMPL
        #--- NOTE the names "stft" and "mel_spec" methods are identical to what is used in the 2 implementation in HiFiGAN code, I keep the same names for easy reference)
        if MEL_IMPL == 'fastpitch':
            self.mel_spec = None
            self.stft = layers.TacotronSTFT(filter_length, hop_length, win_length,n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
        elif MEL_IMPL == 'hifigan':
            #--- *** NOTE *** the following comment was true for models trained up until 2023-05-28 (after that date I made sure the 'hifigan' mel spec is used also when using synthetic wavs.
            #--- (this is the implementation used to calculate mel spec of input on-the-fly during training and validation if we are NOT using synthetic wavs/fine-tuning)
            self.stft = None
            self.mel_spec = partial(mel_spectrogram, 
                                    n_fft = filter_length,
                                    num_mels = n_mel_channels,
                                    sampling_rate = sampling_rate,
                                    hop_size = hop_length, 
                                    win_size = win_length,
                                    fmin = mel_fmin,
                                    fmax = mel_fmax)
        else:
            raise UnknownMelSpecImplementation(f'unknown MEL spec implementation {MEL_SPEC} (should be either "fastpitch" or "hifigan")')
        
        self.initialized = True

    def get_spec(self, audio):
        if not self.initialized:
            raise MelSpecClassNotInitialized('MelSpec class should be initialized using the "init" method before using it')                                             
            
        if self.MEL_IMPL == 'fastpitch':
            return self._get_mel_fastpitch(audio)
        else:
            return self._get_mel_hifigan(audio)
        
    def _get_mel_fastpitch(self, audio):
        melspec = self.stft.mel_spectrogram(audio)          
        return melspec

    def _get_mel_hifigan(self, audio):
        melspec = self.mel_spec(audio)
        return melspec
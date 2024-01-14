from typing import Dict, List

from ssynth.utils.melspec import MelSpec
from sax_synth_inference import load_generator_model

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model

import torch
import torch.nn as nn
from torch import Tensor

class HiFiGANSaxSynth(nn.Module):
    ''' A wrapper of a trained HiFiGAN model, that applies Mel spectrum to input audio, and send the result to the hifigan model
    '''
    def __init__(self, hifigan_model, cfg):
        super().__init__()
        self.hifigan = hifigan_model
        self.hifigan.eval()
        self.melspec = MelSpec()
        self.melspec.init(cfg, 'hifigan')

    def forward(self, x):
        mel = self.melspec.get_spec(x)
        y = self.hifigan(mel)
        return y

class HiFiGANSaxSynthWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "hifigan_sax_synth"

    def get_model_authors(self) -> List[str]:
        return ["Itamar Katz"]

    def get_model_short_description(self) -> str:
        return "Saxophone synthesizer using HiFiGAN"

    def get_model_long_description(self) -> str:
        return "Saxophone synthesizer using HiFiGAN"

    def get_technical_description(self) -> str:
        return "Saxophone synthesizer using HiFiGAN"

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Code": "https://google.com"
        }

    def get_tags(self) -> List[str]:
        return ["sax_synth"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            #NeutoneParameter("min", "min clip threshold", default_value=0.15),
            #NeutoneParameter("max", "max clip threshold", default_value=0.15),
            NeutoneParameter("input_gain", "scale input signal", default_value=1.0),
        ]

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return True

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return True

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [44100]  # Supports all sample rates

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return [1024]  # Supports all buffer sizes

    def aggregate_params(self, params: Tensor) -> Tensor:
        return params  # We want sample-level control, so no aggregation

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        #min_val, max_val, gain = params["min"], params["max"], params["gain"]
        gain = params["input_gain"]
        x = self.model.forward(x) #, min_val, max_val, gain)

        return x

def test_hifigan_sax_synth(use_cuda_if_available = False):
    import numpy as np
    fnm = 'example_models/hifigan_gen_checkpoint_3000.pt'
    print('loading hifigan from checkpoint')
    gen, denoiser, cfg = load_generator_model(fnm, device = 'cpu')
    print('done loading hifigan')
    model = HiFiGANSaxSynth(gen, cfg)

    print('testing using a test siganl...')
    #--- 1 sec of sine-wave @ 44100
    fs = 44100
    x = np.sin(2 * np.pi * 220 * np.arange(fs) / fs).astype(np.float32)
    x = torch.tensor(x).unsqueeze(0)
    if use_cuda_if_available and torch.cuda.is_available():
        print('Using cuda')
        x = x.cuda()
        model = model.cuda()

    y = model(x)
    #--- compare the rms of the signal to expected value
    y_rms = y.square().mean().sqrt().item()
    y_rms_expected = 0.0013820248423144221
    n_digits = 6 # output is different from cpu to gpu
    if np.round(y_rms, n_digits) == np.round(y_rms_expected, n_digits):
        print(f'Output as expected (rounded to {n_digits} digits) :-)')
    else:
        print(f'Output not as expected: y_rms = {y_rms}, expected {y_rms_expected}')

if __name__ == '__main__':
    import pathlib
    from neutone_sdk.utils import save_neutone_model
    
    try:
        print('====== calling the test method ======')
        test_hifigan_sax_synth(use_cuda_if_available = False)
        print('====== done testing ======')
    except Exception as e:
        print(f'Testing failed with error: {e}')
        raise e

    fnm = 'example_models/hifigan_gen_checkpoint_3000.pt'
    print('loading hifigan from checkpoint')
    gen, denoiser, cfg = load_generator_model(fnm, device = 'cpu')
    print('done loading hifigan')
    model = HiFiGANSaxSynth(gen, cfg)
    
    print('Saving to Neutone model')
    w_model = HiFiGANSaxSynthWrapper(model)

    root_dir = pathlib.Path('neutone_export/hifigan')
    save_neutone_model(w_model, root_dir) # , dump_samples = True, submission = True)

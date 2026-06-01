import torch
import torchaudio.transforms as T
from scipy.io import wavfile
import numpy as np

class AudioPreprocessor:
    def __init__(self, sample_rate = 44100, n_fft = 2048, hop_length = 512, max_freq=10000):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.max_freq = max_freq

        freq_res = sample_rate / n_fft
        self.num_bins_to_keep = int(max_freq / freq_res)
        self.window = torch.hann_window(self.win_length)

    def wav_to_tensors(self, path):
        """loads a wav file and return mono tensor"""
        sr, data = wavfile.read(path)
        waveform = torch.from_numpy(data.astype(np.float32)).T

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if sr != self.sample_rate:
            resample = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resample(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform
    
    def waveform_to_complex_stft(self, waveform):
        """Convert waveform to complex STFT."""
        complex_spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            return_complex=True,
            pad_mode='constant'
        )
        return complex_spec[:, :self.num_bins_to_keep, :]
    
    def reconstruct_audio(self, original_complex_spec, soft_mask):
        """Applies the mask and reconstructs the audio via ISTFT"""
        # apply mask to complex spectrogram
        masked_spec = original_complex_spec * soft_mask

        # pad the frequency bins back to full size (n_fft // 2 + 1)
        pad_amount = (self.n_fft // 2 + 1) - masked_spec.shape[1]
        if pad_amount > 0:
            masked_spec = torch.nn.functional.pad(masked_spec, (0, 0, 0, pad_amount))
        
        # Inverse STFT to fetch waveform
        waveform = torch.istft(
            masked_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(masked_spec.device)
        )
        return waveform
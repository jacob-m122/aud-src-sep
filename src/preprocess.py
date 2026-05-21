import torch
import torchaudio
import torchaudio.transforms as T
#from scipy.io import wavfile
import numpy as np

class AudioPreprocessor:
    def __init__(self, sample_rate = 44100, n_fft = 2048, hop_length = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft

        self.stft = T.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            power = None,
            normalized = True
        )


    def wav_to_magnitude_spectrogram(self, path):

        sr, data = wavfile.read(path)
        

        waveform = torch.from_numpy(data.astype(np.float32)).T
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        
        if sr != self.sample_rate:
            resample = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resample(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        complex_spec = self.stft(waveform)
        mag_spec = torch.abs(complex_spec)
        return mag_spec
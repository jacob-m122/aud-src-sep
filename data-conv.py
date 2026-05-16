import os
import torch
import torchaudio
import torchaudio.transforms as T
import kagglehub

path = kagglehub.dataset_download("quanglvitlm/musdb18-hq")

print("Path to dataset files:", path)

class AudioPreprocessor:
    def __init__(self, sample_rate = 44100, n_fft = 2048, hop_length = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft

        self.stft = T.spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            power = None,
            normalized = True
        )
        self.gl = T.GriffinLim(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            power = 1,
            normalized = True
        )
    
    def wave_to_mag_spect(self, path):
        waveform, sr = torch.load(path)

        if sr != self.sample_rate:
            resample = T.Resample(og_freq=sr, new_freq=self.sample_rate)
            waveform = resample(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)






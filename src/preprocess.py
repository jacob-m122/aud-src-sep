import torch
import torchaudio
import torchaudio.transforms as T
import kagglehub

torchaudio.set_audio_backend("soundfile")

class AudioPreprocessor:
    def __init__(self, sample_rate = 44100, n_fft = 2048, hop_length = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft

        # FIX: Capital 'S' in Spectrogram
        self.stft = T.Spectrogram(
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
            #normalized = True
        )
    
    # FIX: Renamed to match the method call in dataloader.py
    def wav_to_magnitude_spectrogram(self, path):
        # FIX: Use torchaudio.load instead of torch.load (which is for .pt models)
        waveform, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            # FIX: The argument is 'orig_freq', not 'og_freq'
            resample = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resample(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        complex_spec = self.stft(waveform)
        mag_spec = torch.abs(complex_spec)

        return mag_spec
    
    def mag_spec_to_wave(self, mag_spec, output_path):
        re_waveform = self.gl(mag_spec)
        torchaudio.save(output_path, re_waveform, self.sample_rate)
        print(f"Saved reconstructed audio to: {output_path}")

if __name__ == "__main__":
    preprocessor = AudioPreprocessor(sample_rate=44100, n_fft=2048, hop_length=512)
"""Dataloader.py"""
import glob
import torch
import os
from torch.utils.data import Dataset, DataLoader
import itertools
from preprocess import AudioPreprocessor

class MusdbDataset(Dataset):
    def __init__(self, dataset_root, segment_length=256, sample_rate=44100, n_fft=2048, hop_length=512):
        self.dataset_root = dataset_root
        self.segment_samples = segment_length * hop_length
        self.processor = AudioPreprocessor(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length)
        self.track_pairs = self._build_dataset_index()

   
    def _build_dataset_index(self):
        pairs = []

        song_folders = [f.path for f in os.scandir(self.dataset_root) if f.is_dir()]

        stems = ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']
        
        for song_dir in song_folders:
            for primary_stem, bleed_stem in itertools.permutations(stems, 2):
                primary_path = os.path.join(song_dir, primary_stem)
                bleed_path = os.path.join(song_dir, bleed_stem)

                if os.path.exists(primary_path) and os.path.exists(bleed_path):
                    pairs.append({
                        'primary_path': primary_path,
                        'bleed_path': bleed_path
                    })

        if len(pairs) == 0:
            print("error: no track pairs found.")

        return pairs
    
    def _augment_waveform(self, primary_wav, reference_wav):

        bleed_gain = torch.rand(1).item() * 0.35 + 0.05

        phase_flip = -1.0 if torch.rand(1).item() > 0.5 else 1.0
        bleed_signal = reference_wav * bleed_gain * phase_flip



        delay_seconds = torch.rand(1).item() * 0.99 + 0.01
        delay_samples = int(self.processor.sample_rate * delay_seconds)
        

        decay_factor = torch.rand(1).item() * 0.5 + 0.1
        
        reflection = torch.zeros_like(primary_wav)
        if primary_wav.shape[-1] > delay_samples:
            reflection[:, delay_samples:] = primary_wav[:, :-delay_samples] * decay_factor


        noise_level = torch.rand(1).item() * 0.009 + 0.001
        noise_floor = torch.randn_like(primary_wav) * noise_level


        artifacted_primary = primary_wav + bleed_signal + reflection + noise_floor

        return artifacted_primary

    def __len__(self):
        return len(self.track_pairs)
    
    def __getitem__(self, id):
        pair = self.track_pairs[id]

        clean_primary_wav = self.processor.wav_to_tensors(pair['primary_path'])
        reference_wav = self.processor.wav_to_tensors(pair['bleed_path'])

        min_len = min(clean_primary_wav.shape[-1], reference_wav.shape[-1])

        if min_len > self.segment_samples:
            start = torch.randint(0, min_len - self.segment_samples, (1,)).item()
            clean_primary_wav = clean_primary_wav[:, start:start + self.segment_samples]
            reference_wav = reference_wav[:, start:start + self.segment_samples]
        else:
            pad_amount = self.segment_samples - min_len
            clean_primary_wav = torch.nn.functional.pad(clean_primary_wav[:, :min_len], (0, pad_amount))
            reference_wav = torch.nn.functional.pad(reference_wav[:, :min_len], (0, pad_amount))
        

        artifacted_primary_wav = self._augment_waveform(clean_primary_wav, reference_wav)

        max_val = torch.max(torch.abs(artifacted_primary_wav))
        if max_val > 1.0:
            artifacted_primary_wav = artifacted_primary_wav / max_val
            clean_primary_wav = clean_primary_wav / max_val
            reference_wav = reference_wav / max_val


        return artifacted_primary_wav, reference_wav, clean_primary_wav

import glob
import torch
import os
from torch.utils.data import Dataset, DataLoader
from preprocess import AudioPreprocessor

class MedleyDbDataset(Dataset):
    def __init__(self, dataset_root, segment_length=256, sample_rate=44100, n_fft=2048, hop_length=512):
        self.dataset_root = dataset_root
        self.segment_samples = segment_length * hop_length
        self.processor = AudioPreprocessor(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length)
        self.track_pairs = self._build_dataset_index()

   
    def _build_dataset_index(self):
        pairs = []
        
        song_folders = [f.path for f in os.scandir(self.dataset_root) if f.is_dir()]

        for song_dir in song_folders:
            multitrack_dir = os.path.join(song_dir, f"{os.path.basename(song_dir)}_MULTITRACKS")
            if not os.path.exists(multitrack_dir):
                continue

            # Ensure consistent pairing by sorting the files
            wav_files = sorted(glob.glob(os.path.join(multitrack_dir, "*.wav")))

            for i in range(len(wav_files) - 1):
                pairs.append({
                    'primary_path': wav_files[i],
                    'bleed_path': wav_files[i+1]
                })
        return pairs
    
    def _augment_waveform(self, primary_wav, reference_wav):
        """Injects synthetic artifacting: bleed, reflections, noise floor"""
        #simulate mic bleed (10% to 30% of the reference signal
        bleed_gain = torch.rand(1).item() * 0.2 + 0.1
        bleed_signal = reference_wav * bleed_gain

        #simulate room reflections (50ms delay with decay)
        delay_samples = int(self.processor.sample_rate * 0.65)
        decay_factor = 0.4
        reflection = torch.zeros_like(primary_wav)
        if primary_wav.shape[-1] > delay_samples:
            reflection[:, delay_samples:] = primary_wav[:, :-delay_samples] * decay_factor

        noise_floor = torch.randn_like(primary_wav) * 0.005

        artifacted_primary = primary_wav + bleed_signal + reflection + noise_floor

        #clipping prevention
        max_val = torch.max(torch.abs(artifacted_primary))
        if max_val > 1.0:
            artifacted_primary = artifacted_primary / max_val
        return artifacted_primary

    def __len__(self):
        return len(self.track_pairs)
    
    def __getitem__(self, id):
        pair = self.track_pairs[id]

        #load raw waveforms
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
        
        # generate corrupted input
        artifacted_primary_wav = self._augment_waveform(clean_primary_wav, reference_wav)

        # convert to complex STFT
        complex_artifacted = self.processor.waveform_to_complex_stft(artifacted_primary_wav)
        complex_reference = self.processor.waveform_to_complex_stft(reference_wav)
        complex_clean_target = self.processor.waveform_to_complex_stft(clean_primary_wav)

        # extract magnitudes for model
        mag_artifacted = torch.abs(complex_artifacted)
        mag_reference = torch.abs(complex_reference)
        mag_clean_target = torch.abs(complex_clean_target)

        # return inputs (magnitudes) and the target mask (clean / corrupted ratio)
        target_mask = mag_clean_target / (mag_artifacted + 1e-8)
        target_mask = torch.clamp(target_mask, 0.0, 1.0)

        return mag_artifacted, mag_reference, target_mask, complex_artifacted


#if __name__ == "__main__":
#    DATASET_PATH = "./data/sample/TestSong"
#
#    if os.path.exists(DATASET_PATH):
#        dataset = MedleyDbDataset(dataset_root=DATASET_PATH, segment_length=256)
#        print(f"Total track pairs indexed: {len(dataset)}")
#
#        
#        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
#
#        for batch_id, (batch_track_a, batch_track_b) in enumerate(dataloader):
#            print(f"Batch {batch_id} loaded successfully")
#            
#            
#            print(f"Track A batch shape: {batch_track_a.shape}")
#            print(f"Track B batch shape: {batch_track_b.shape}")
#            break
#    else:
#        print("Directory not found. Please verify the relative path.")
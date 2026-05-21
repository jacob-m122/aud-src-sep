import glob
import torch
import os
from torch.utils.data import Dataset, DataLoader
from preprocess import AudioPreprocessor

class MedleyDbDataset(Dataset):
    def __init__(self, dataset_root, segment_length=256, sample_rate=44100, n_fft=2048, hop_length=512):
        self.dataset_root = dataset_root
        self.segment_length = segment_length
        self.processor = AudioPreprocessor(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length)

        # FIX: Call the exact method name defined below
        self.track_pairs = self._build_dataset_index()

    # FIX: Added leading underscore to match the __init__ call
    def _build_dataset_index(self):
        pairs = []

        song_folders = [f.path for f in os.scandir(self.dataset_root) if f.is_dir()]

        for song_dir in song_folders:
            multitrack_dir = os.path.join(song_dir, f"{os.path.basename(song_dir)}_MULTITRACKS")
            if not os.path.exists(multitrack_dir):
                continue

            # FIX: Extension is .wav, not .wave
            wav_files = glob.glob(os.path.join(multitrack_dir, "*.wav"))

            for i in range(len(wav_files) - 1):
                pairs.append({
                    'primary_path': wav_files[i],
                    'bleed_path': wav_files[i+1]
                })

        if len(pairs) == 0:
            print("MedleyDB structure not found. Falling back to flat directory search...")
            # FIX: Use recursive=True and **/*.wav so it finds files buried inside 'TestSong/Violet_Wave/'
            search_pattern = os.path.join(self.dataset_root, "**", "*.wav")
            wav_files = sorted(glob.glob(search_pattern, recursive=True))
            
            for i in range(len(wav_files) - 1):
                pairs.append({
                    'primary_path': wav_files[i],
                    'bleed_path': wav_files[i+1]
                })
    
        return pairs
    
    def __len__(self):
        return len(self.track_pairs)
    
    def __getitem__(self, id):
        pair = self.track_pairs[id]

        # 1. Dynamically compute magnitude spectrograms on the fly
        spec_a = self.processor.wav_to_magnitude_spectrogram(pair['primary_path'])
        spec_b = self.processor.wav_to_magnitude_spectrogram(pair['bleed_path'])
        
        # 2. Slice to a fixed chunk size so PyTorch can batch them
        if spec_a.shape[-1] > self.segment_length:
            max_start = spec_a.shape[-1] - self.segment_length
            start = torch.randint(0, max_start, (1,)).item()
            
            spec_a = spec_a[:, :, start:start + self.segment_length]
            spec_b = spec_b[:, :, start:start + self.segment_length]

        else:
            pad_amount = self.segment_length - spec_a.shape[-1]
            spec_a = torch.nn.functional.pad(spec_a, (0, pad_amount))
            spec_b = torch.nn.functional.pad(spec_b, (0, pad_amount))
            
        return spec_a, spec_b


# FIX: Proper Python main execution block syntax
if __name__ == "__main__":
    DATASET_PATH = "./data/sample/TestSong"

    if os.path.exists(DATASET_PATH):
        dataset = MedleyDbDataset(dataset_root=DATASET_PATH, segment_length=256)
        print(f"Total track pairs indexed: {len(dataset)}")

        # FIX: Dataloader creation must be indented inside the IF block, otherwise it crashes if folder is missing.
        # Note: num_workers=0 is safer for testing on local environments to prevent multiprocessing hangs.
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        for batch_id, (batch_track_a, batch_track_b) in enumerate(dataloader):
            print(f"Batch {batch_id} loaded successfully")
            
            # FIX: Added .shape. Printing the raw tensor will flood the terminal with millions of numbers.
            print(f"Track A batch shape: {batch_track_a.shape}")
            print(f"Track B batch shape: {batch_track_b.shape}")
            break # Stop after one batch to prove the logic works
    else:
        print("Directory not found. Please verify the relative path.")
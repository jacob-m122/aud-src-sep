import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import kagglehub

class AudioStemDataset(Dataset):
    def __init__(self, file_paths, sample_rate=44100, n_fft=2048):
        self.file_paths = file_paths
        self.n_fft = n_fft

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pass
    
class CrossTrackAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossTrackAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

def forward(self, track_a, track_b):
    q = self.query(track_a)
    k = self.key(track_b)
    v = self.value(track_b)

    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    soft_attn = F.softmax(attn_weights, dim=-1)

    bleed_estimation = torch.matmul(soft_attn, v)

    refined_signal = track_a - bleed_estimation

    return refined_signal

class AntiArtifactModel(nn.Module):
    def __init__(self, embed_dim = 128):
        super(AntiArtifactModel, self).__init__()

        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU())
        self.attention = CrossTrackAttention(embed_dim=128)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(16, 1, 3), nn.Sigmoid())


        self.feature_projection = nn.Linear(16, embed_dim)
        self.attention = CrossTrackAttention(embed_dim=embed_dim)
        self.feature_reconstruction = nn.Linear(16, embed_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1)
        
        )

    

    def forward(self, primary, reference):
        enc_primary = self.encoder(primary)
        enc_reference = self.encoder(reference)

        B, C, F, T = enc_primary.shape
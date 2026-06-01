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

        #layer1: encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, kernel_size=3, padding=1), 
            nn.ReLU())
        
        #layer2: attention mechanism
        self.attention = CrossTrackAttention(embed_dim=embed_dim)
        self.feature_projection = nn.Linear(16, embed_dim)
        self.feature_reconstruction = nn.Linear(embed_dim, 16)

        #layer3: decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        
        )

    
    def forward(self, primary, reference):
        enc_primary = self.encoder(primary)
        enc_reference = self.encoder(reference)

        B, C, F, T = enc_primary.shape

        flat_primary = enc_primary.view(B, C, F * T).permute(0, 2, 1)
        flat_reference = enc_reference.view(B, C, F * T).permute(0, 2, 1)

        proj_primary = self.feature_projection(flat_primary)
        proj_reference = self.feature_projection(flat_reference)

        attn_out = self.attention(proj_primary, proj_reference)

        recon_flat = self.feature_reconstruction(attn_out)
        recon_4d = recon_flat.permute(0, 2, 1).view(B, C, F, T)

        output = self.decoder(recon_4d)
        return output


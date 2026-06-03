import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import AudioPreprocessor
from model import AntiArtifactModel
from dataloader import MusdbDataset
from torchmetrics.audio import SignalDistortionRatio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

TRAIN_PATH = "./data/musdb18_hq/train"
VAL_PATH = "./data/musdb18_hq/test"

train_dataset = MusdbDataset(dataset_root=TRAIN_PATH)
val_dataset = MusdbDataset(dataset_root=VAL_PATH)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AntiArtifactModel(embed_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

criterion = nn.L1Loss()


def calculate_lsd(mag_clean, mag_recon):
    log_clean = 20 * torch.log10(mag_clean + 1e-8)
    log_recon = 20 * torch.log10(mag_recon + 1e-8)

    diff = (log_clean-log_recon) ** 2
    lsd = torch.mean(torch.sqrt(torch.mean(diff,dim=1)))
    return lsd.item()

def validate_and_evaluate(model, val_dataloader):
    model.eval()

    sdr_metric = SignalDistortionRatio().to(device)
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
    total_lsd = 0.0

    with torch.no_grad():
        for mag_artifacted, mag_reference, target_mask, complex_artifacted in val_dataloader:
            mag_artifacted = mag_artifacted.to(device)
            mag_reference = mag_reference.to(device)
            complex_artifacted = complex_artifacted.to(device)

            predicted_mask = model(mag_artifacted, mag_reference)

            processor = AudioPreprocessor(sample_rate=44100, n_fft=2048)
            recon_audio = processor.reconstruct_audio(complex_artifacted, predicted_mask)

            clean_audio = processor.reconstruct_audio(complex_artifacted, target_mask)

            #calculate SIR/SDR
            sdr_metric.update(recon_audio, clean_audio)
            si_sdr_metric.update(recon_audio, clean_audio)

            #calculate lsd
            mag_recon = torch.abs(processor.waveform_to_complex_stft(recon_audio)).to(device)
            mag_clean_true = mag_artifacted * target_mask

            total_lsd += calculate_lsd(mag_clean_true, mag_recon)

    print(f"Validation SDR: {sdr_metric.compute():.2f} dB")
    print(f"Validation SI-SDR: {si_sdr_metric.compute():.2f} dB")
    print(f"Validation Mean LSD: {total_lsd / len(val_dataloader):.4f}")

    sdr_metric.reset()
    si_sdr_metric.reset()

def train_loop(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=50):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for _, (mag_artifacted, mag_reference, target_mask, _) in enumerate(train_dataloader):
            mag_artifacted = mag_artifacted.to(device)
            mag_reference = mag_reference.to(device)
            target_mask = target_mask.to(device)

            optimizer.zero_grad()

            predicted_mask = model(mag_artifacted, mag_reference)

            loss = criterion(predicted_mask, target_mask)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss {train_loss/len(train_dataloader): .4f}")
        
        if (epoch + 1) % 5 == 0:
            validate_and_evaluate(model, val_dataloader)
    

if __name__ == "__main__":

    train_loop(model, train_dataloader, val_dataloader, optimizer, criterion)
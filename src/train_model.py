import os
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

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Colab gpu detected: {torch.cuda.get_device_name(0)} ")
else:
    device = torch.device("cpu")
    print("no gpu detected")



model = AntiArtifactModel(embed_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

processor = AudioPreprocessor(sample_rate=44100, n_fft=2048)


def calculate_lsd(mag_clean, mag_recon):
    log_clean = 20 * torch.log10(mag_clean + 1e-8)
    log_recon = 20 * torch.log10(mag_recon + 1e-8)

    diff = (log_clean-log_recon) ** 2
    lsd = torch.mean(torch.sqrt(torch.mean(diff,dim=1)))
    return lsd.item()

def validate_and_evaluate(model, val_dataloader, criterion):
    model.eval()
    sdr_metric = SignalDistortionRatio().to(device)
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
    total_lsd = 0.0

    total_val_loss = 0.0

    with torch.no_grad():
        for artifacted_wav, reference_wav, clean_wav in val_dataloader:
            artifacted_wav = artifacted_wav.to(device)
            reference_wav = reference_wav.to(device)
            clean_wav = clean_wav.to(device)

            # Compute STFTs dynamically on the hardware accelerator
            complex_artifacted = processor.waveform_to_complex_stft(artifacted_wav)
            complex_reference = processor.waveform_to_complex_stft(reference_wav)
            complex_clean = processor.waveform_to_complex_stft(clean_wav)

            mag_artifacted = torch.abs(complex_artifacted).unsqueeze(1)
            mag_reference = torch.abs(complex_reference).unsqueeze(1)
            mag_clean = torch.abs(complex_clean).unsqueeze(1)
            
            target_mask = torch.clamp(mag_clean / (mag_artifacted + 1e-8), 0.0, 1.0)
            predicted_mask = model(mag_artifacted, mag_reference)

            loss = criterion(predicted_mask, target_mask)
            total_val_loss += loss.item()

            recon_audio = processor.reconstruct_audio(complex_artifacted, predicted_mask)
            clean_audio = processor.reconstruct_audio(complex_artifacted, target_mask)
            #dithering to prevent 
            safe_recon = recon_audio + 1e-7 * torch.randn_like(recon_audio)
            safe_clean = clean_audio + 1e-7 * torch.randn_like(clean_audio)
            #calculate SIR/SDR
            sdr_metric.update(safe_recon, safe_clean)
            si_sdr_metric.update(safe_recon, safe_clean)

            #calculate lsd
            mag_recon = torch.abs(processor.waveform_to_complex_stft(recon_audio)).to(device)
            mag_clean_true = mag_artifacted * target_mask

            total_lsd += calculate_lsd(mag_clean_true, mag_recon)

    print(f"Validation Loss: {total_val_loss / len(val_dataloader):.4f}")
    print(f"Validation SDR: {sdr_metric.compute():.2f} dB")
    print(f"Validation SI-SDR: {si_sdr_metric.compute():.2f} dB")
    print(f"Validation Mean LSD: {total_lsd / len(val_dataloader):.4f}")

    sdr_metric.reset()
    si_sdr_metric.reset()


def train_loop(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_id, (artifacted_wav, reference_wav, clean_wav) in enumerate(train_dataloader):

            artifacted_wav = artifacted_wav.to(device)
            reference_wav = reference_wav.to(device)
            clean_wav = clean_wav.to(device)

            # Perform STFT math ON THE ACCELERATOR
            complex_artifacted = processor.waveform_to_complex_stft(artifacted_wav)
            complex_reference = processor.waveform_to_complex_stft(reference_wav)
            complex_clean = processor.waveform_to_complex_stft(clean_wav)

            mag_artifacted = torch.abs(complex_artifacted).unsqueeze(1)
            mag_reference = torch.abs(complex_reference).unsqueeze(1)
            mag_clean = torch.abs(complex_clean).unsqueeze(1)

            # Generate target mask
            target_mask = torch.clamp(mag_clean / (mag_artifacted + 1e-8), 0.0, 1.0)

            optimizer.zero_grad()
            predicted_mask = model(mag_artifacted, mag_reference)
            loss = criterion(predicted_mask, target_mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_id % 10 == 0:
                print(f" -> Batch {batch_id}/{len(train_dataloader)} | Current Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{epochs} | Loss {train_loss/len(train_dataloader): .4f}")
        
        if (epoch + 1) % 5 == 0:
            validate_and_evaluate(model, val_dataloader, criterion)
    

if __name__ == "__main__":

    train_loop(model, train_dataloader, val_dataloader, optimizer, criterion)

    print("Training complete. Saving model weights...")
    save_path = '/content/drive/MyDrive/aud-src-sep/anti_artifact_model_final.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to: {save_path}")
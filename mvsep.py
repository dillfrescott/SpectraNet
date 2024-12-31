import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prodigyopt import Prodigy
import numpy as np

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention weights
        att = self.conv(x)
        att = self.sigmoid(att)
        # Apply attention weights
        out = x * att
        return out

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        se = F.relu(self.fc1(se))
        se = self.fc2(se)
        return x * self.sigmoid(se)

# U-Net-like CNN architecture
class UNetCNN(nn.Module):
    def __init__(self, in_channels=2, hidden_size=512, num_layers=5, dilation_rate=2):
        super(UNetCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dilation_rate = dilation_rate

        # First layer
        self.conv1 = nn.Conv2d(in_channels, hidden_size, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.att1 = SpatialAttention(hidden_size)

        # Encoder layers
        self.encoder_convs = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()
        self.encoder_atts = nn.ModuleList()
        self.dilation_rates = [dilation_rate ** i for i in range(num_layers - 1)]
        for dilation in self.dilation_rates:
            conv = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=(3, 3),
                padding=(dilation, dilation), dilation=(dilation, dilation)
            )
            bn = nn.BatchNorm2d(hidden_size)
            att = SpatialAttention(hidden_size)
            self.encoder_convs.append(conv)
            self.encoder_bns.append(bn)
            self.encoder_atts.append(att)

        # Decoder layers
        self.decoder_convs = nn.ModuleList()
        self.decoder_bns = nn.ModuleList()
        self.decoder_atts = nn.ModuleList()
        for dilation in reversed(self.dilation_rates):
            conv = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=(3, 3),
                padding=(dilation, dilation), dilation=(dilation, dilation)
            )
            bn = nn.BatchNorm2d(hidden_size)
            att = SpatialAttention(hidden_size)
            self.decoder_convs.append(conv)
            self.decoder_bns.append(bn)
            self.decoder_atts.append(att)

        # Last layer for instrumental and vocal magnitude
        self.conv_inst = nn.Conv2d(hidden_size, in_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv_vocal = nn.Conv2d(hidden_size, in_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        # First layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.att1(x)

        # Encoder layers
        encoder_outputs = []
        for conv, bn, att in zip(self.encoder_convs, self.encoder_bns, self.encoder_atts):
            residual = x
            x = F.relu(bn(conv(x)))
            x = att(x)
            x = x + residual
            encoder_outputs.append(x)

        # Decoder layers
        for i, (conv, bn, att) in enumerate(zip(self.decoder_convs, self.decoder_bns, self.decoder_atts)):
            x = x + encoder_outputs[-i-1]  # Skip connection
            residual = x
            x = F.relu(bn(conv(x)))
            x = att(x)
            x = x + residual

        # Predict instrumental and vocal magnitude
        inst_mag = self.conv_inst(x)
        vocal_mag = self.conv_vocal(x)

        return inst_mag, vocal_mag

def loss_fn(pred_inst_mag, pred_vocal_mag, target_inst_mag, target_vocal_mag):
    # L1 loss for both instrumental and vocal magnitudes
    inst_loss = torch.mean(torch.abs(pred_inst_mag - target_inst_mag))  # L1 loss for instrument
    vocal_loss = torch.mean(torch.abs(pred_vocal_mag - target_vocal_mag))  # L1 loss for vocals
    return inst_loss + vocal_loss

# Custom Dataset class
class MUSDBDataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_length=264600, n_fft=4096, hop_length=1024, segment=True):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment = segment
        self.tracks = [os.path.join(root_dir, track) for track in os.listdir(root_dir)]
        self.window = torch.hann_window(n_fft)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_path = self.tracks[idx]
        instrumental, _ = torchaudio.load(os.path.join(track_path, 'other.wav'))
        vocal, _ = torchaudio.load(os.path.join(track_path, 'vocals.wav'))

        # Ensure both signals have the same number of channels
        if instrumental.shape[0] != 2 or vocal.shape[0] != 2:
            raise ValueError("Audio files must have 2 channels.")

        # Ensure all signals have the same length
        min_length = min(instrumental.shape[1], vocal.shape[1])
        instrumental = instrumental[:, :min_length]
        vocal = vocal[:, :min_length]

        # Create the mixture by summing instrumental and vocal
        mixture = instrumental + vocal

        # Convert to spectrograms with Hann window
        mixture_spec = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        instrumental_spec = torch.stft(instrumental, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        vocal_spec = torch.stft(vocal, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)

        # Convert to magnitude spectrograms
        mixture_spec = torch.abs(mixture_spec)
        instrumental_spec = torch.abs(instrumental_spec)
        vocal_spec = torch.abs(vocal_spec)

        # Optionally segment the signals
        if self.segment and self.segment_length:
            if mixture_spec.shape[2] >= self.segment_length // self.hop_length:
                start = torch.randint(0, mixture_spec.shape[2] - self.segment_length // self.hop_length, (1,))
                mixture_spec = mixture_spec[:, :, start:start + self.segment_length // self.hop_length]
                instrumental_spec = instrumental_spec[:, :, start:start + self.segment_length // self.hop_length]
                vocal_spec = vocal_spec[:, :, start:start + self.segment_length // self.hop_length]
            else:
                mixture_spec = F.pad(mixture_spec, (0, self.segment_length // self.hop_length - mixture_spec.shape[2]))
                instrumental_spec = F.pad(instrumental_spec, (0, self.segment_length // self.hop_length - instrumental_spec.shape[2]))
                vocal_spec = F.pad(vocal_spec, (0, self.segment_length // self.hop_length - vocal_spec.shape[2]))

        return mixture_spec, instrumental_spec, vocal_spec, mixture, instrumental, vocal

# Training function
def train(model, dataloader, optimizer, scheduler, loss_fn, device, epochs, checkpoint_steps, checkpoint_path=None):
    model.to(device)
    step = 0
    avg_loss = 0.0
    loss_log = []
    progress_bar = tqdm(total=epochs * len(dataloader))

    if checkpoint_path:
        # Use weights_only=True when loading the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        avg_loss = checkpoint['avg_loss']
        loss_log = checkpoint['loss_log']
        print(f"Resuming training from step {step} with average loss {avg_loss:.4f}")

    model.train()
    for epoch in range(epochs):
        for mixture_spec, instrumental_spec, vocal_spec, _, _, _ in dataloader:
            mixture_spec = mixture_spec.to(device)
            instrumental_spec = instrumental_spec.to(device)
            vocal_spec = vocal_spec.to(device)

            target_inst_mag = instrumental_spec
            target_vocal_mag = vocal_spec

            optimizer.zero_grad()

            pred_inst_mag, pred_vocal_mag = model(mixture_spec)
            loss = loss_fn(pred_inst_mag, pred_vocal_mag, target_inst_mag, target_vocal_mag)

            loss.backward()

            # Check for NaN values in gradients
            has_nan = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan = True
                    break

            if has_nan:
                if step > 0:
                    print("NaN detected in gradients, skipping this batch.")
                optimizer.zero_grad()
                continue

            optimizer.step()
            scheduler.step()

            avg_loss = (avg_loss * step + loss.item()) / (step + 1)
            loss_log.append(loss.item())
            step += 1
            progress_bar.update(1)
            desc = f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Avg Loss: {avg_loss:.4f}"

            if step % checkpoint_steps == 0:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'loss_log': loss_log
                }, f"checkpoint_step_{step}.pt")

            progress_bar.set_description(desc)

    torch.save({'loss_log': loss_log}, 'loss_log.pt')
    progress_bar.close()

def inference(model, checkpoint_path, input_wav_path, output_instrumental_path, output_vocal_path,
              chunk_size=88200, overlap=44100, device='cpu', n_fft=4096, hop_length=1024):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)  # Set weights_only=True
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model.to(device)

    input_audio, sr = torchaudio.load(input_wav_path)
    if input_audio.shape[0] != 2:
        raise ValueError("Input audio must have 2 channels.")
    input_audio = input_audio.to(device)

    total_length = input_audio.shape[1]
    num_chunks = (total_length - overlap) // (chunk_size - overlap)
    instrumentals = torch.zeros_like(input_audio)
    vocals = torch.zeros_like(input_audio)

    cross_fade_length = overlap // 2
    window = torch.hann_window(n_fft).to(device)

    with tqdm(total=num_chunks, desc="Processing audio") as pbar:
        for i in range(0, total_length - chunk_size + 1, chunk_size - overlap):
            chunk = input_audio[:, i:i + chunk_size]

            chunk_spec = torch.stft(chunk, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            chunk_mag = torch.abs(chunk_spec)
            chunk_phase = torch.angle(chunk_spec)

            chunk_mag = chunk_mag.unsqueeze(0)

            with torch.no_grad():
                pred_inst_mag, pred_vocal_mag = model(chunk_mag)

            pred_inst_mag = pred_inst_mag.squeeze(0)
            pred_vocal_mag = pred_vocal_mag.squeeze(0)

            # Subtract the predicted vocal magnitude from the mixture magnitude
            inst_spec_mag = chunk_mag - pred_vocal_mag
            inst_spec_mag = torch.clamp(inst_spec_mag, min=0.0)

            inst_spec = inst_spec_mag * torch.exp(1j * chunk_phase)

            # Ensure inst_spec has the correct shape for istft
            inst_spec = inst_spec.squeeze(0)

            inst_chunk = torch.istft(inst_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=chunk_size, return_complex=False)

            # Compute vocals by subtracting the instrumental from the original input
            vocal_chunk = chunk - inst_chunk

            if i == 0:
                # For the first chunk, don't apply cross-fading
                instrumentals[:, i:i + chunk_size] = inst_chunk
                vocals[:, i:i + chunk_size] = vocal_chunk
            else:
                # Apply cross-fading for subsequent chunks
                fade_in = torch.linspace(0, 1, cross_fade_length).to(device)
                fade_out = torch.linspace(1, 0, cross_fade_length).to(device)

                inst_chunk[:, :cross_fade_length] *= fade_in
                instrumentals[:, i:i + cross_fade_length] *= fade_out
                instrumentals[:, i:i + cross_fade_length] += inst_chunk[:, :cross_fade_length]

                vocal_chunk[:, :cross_fade_length] *= fade_in
                vocals[:, i:i + cross_fade_length] *= fade_out
                vocals[:, i:i + cross_fade_length] += vocal_chunk[:, :cross_fade_length]

            instrumentals[:, i + cross_fade_length:i + chunk_size] = inst_chunk[:, cross_fade_length:]
            vocals[:, i + cross_fade_length:i + chunk_size] = vocal_chunk[:, cross_fade_length:]

            pbar.update(1)

    instrumentals = torch.clamp(instrumentals, -1.0, 1.0)
    vocals = torch.clamp(vocals, -1.0, 1.0)

    torchaudio.save(output_instrumental_path, instrumentals.cpu(), sr)
    torchaudio.save(output_vocal_path, vocals.cpu(), sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model for instrumental separation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='train', help='Path to training dataset')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--checkpoint_steps', type=int, default=2000, help='Save checkpoint every X steps')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input WAV file for inference')
    parser.add_argument('--output_instrumental', type=str, default='output_instrumental.wav', help='Path to output instrumental WAV file')
    parser.add_argument('--output_vocal', type=str, default='output_vocal.wav', help='Path to output vocal WAV file')
    parser.add_argument('--segment_length', type=int, default=264600, help='Segment length for training')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers in the CNN model')
    parser.add_argument('--n_fft', type=int, default=4096, help='Number of FFT bins for STFT')
    parser.add_argument('--hop_length', type=int, default=1024, help='Hop length for STFT')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetCNN(in_channels=2, hidden_size=256, num_layers=args.num_layers)
    optimizer = Prodigy(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    if args.train:
        train_dataset = MUSDBDataset(root_dir=args.data_dir, segment_length=args.segment_length, n_fft=args.n_fft, hop_length=args.hop_length, segment=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)

        total_steps = args.epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        train(model, train_dataloader, optimizer, scheduler, loss_fn, device, args.epochs, args.checkpoint_steps, checkpoint_path=args.checkpoint_path)
    elif args.infer:
        if args.input_wav is None:
            print("Please specify an input WAV file for inference using --input_wav")
            return
        model = UNetCNN(in_channels=2, hidden_size=256, num_layers=args.num_layers)
        inference(model, args.checkpoint_path, args.input_wav, args.output_instrumental, args.output_vocal, device=device, n_fft=args.n_fft, hop_length=args.hop_length)
    else:
        print("Please specify either --train or --infer")

if __name__ == '__main__':
    main()

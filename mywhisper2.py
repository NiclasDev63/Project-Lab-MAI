import torch
import whisper
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
class AudioFeatureDataset(Dataset):
    def __init__(self, num_samples, max_audio_length=480000):
        self.num_samples = num_samples
        self.max_audio_length = max_audio_length 
        
        base_model = whisper.load_model("large-v3")
        self.n_mels = base_model.dims.n_mels
        self.n_audio_state = base_model.dims.n_audio_state
        del base_model
        
    def __len__(self):
        return self.num_samples
    
    def generate_random_mel(self):
        time_frames = 3000  
        return torch.randn(self.n_mels, time_frames)
    
    def generate_target_embeddings(self):
        time_frames = 1500 
        return torch.randn(time_frames, self.n_audio_state)
    
    def __getitem__(self, idx):
        mel_features = self.generate_random_mel()
        
        target_embeddings = self.generate_target_embeddings()
        
        return {
            "mel_features": mel_features,
            "target_embeddings": target_embeddings
        }

def collate_fn(batch):
    mel_features = [item["mel_features"] for item in batch]
    target_embeddings = [item["target_embeddings"] for item in batch]
    
    max_mel_length = max(mel.shape[1] for mel in mel_features)
    padded_mels = torch.zeros(len(batch), mel_features[0].shape[0], max_mel_length)
    for i, mel in enumerate(mel_features):
        padded_mels[i, :, :mel.shape[1]] = mel
    
    max_embed_length = max(embed.shape[0] for embed in target_embeddings)
    padded_embeds = torch.zeros(len(batch), max_embed_length, target_embeddings[0].shape[1])
    for i, embed in enumerate(target_embeddings):
        padded_embeds[i, :embed.shape[0], :] = embed
    
    return {
        "mel_features": padded_mels,
        "target_embeddings": padded_embeds
    }

class WhisperEncoderTrainer:
    def __init__(self, learning_rate=1e-5):
        print("Initializing Whisper encoder trainer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("large-v3").to(device)
        
        self.encoder = self.model.encoder
        self.encoder.to(self.device)
        # i think we dont need this? Maybe we do for training but this can be undone easily
        del self.model.decoder
        
        self.optimizer = AdamW(self.encoder.parameters(), lr=learning_rate)
        print("Initialization complete!")
    
    def compute_loss(self, encoder_output, target_embeddings):
        """
        Compute MSE loss between encoder output and target embeddings
        You might want to use a different loss function depending on your specific needs
        """
        return torch.nn.functional.mse_loss(encoder_output, target_embeddings)
    
    def train_epoch(self, dataloader, epoch):
        self.encoder.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                mel_features = batch["mel_features"].to(self.device)
                target_embeddings = batch["target_embeddings"].to(self.device)
                last_time = time.time()
                encoder_output = self.encoder(mel_features)
                print("Encoder output shape:", encoder_output.shape)
                current_time = time.time()
                step_duration = current_time - last_time
                last_time = current_time
                print(f"Step took {step_duration:.4f} seconds")
                
                
                loss = self.compute_loss(encoder_output, target_embeddings)
                print(loss)  
                current_time = time.time()
                step_duration = current_time - last_time
                last_time = current_time
                print(f"Step took {step_duration:.4f} seconds")

                self.optimizer.zero_grad()
                loss.backward()
                print("Backward pass complete")
                current_time = time.time()
                step_duration = current_time - last_time
                last_time = current_time
                print(f"Step took {step_duration:.4f} seconds")

                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / len(dataloader)
    
    def train(self, num_epochs=10, batch_size=8, num_samples=1000):
        dataset = AudioFeatureDataset(num_samples=num_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            try:
                avg_loss = self.train_epoch(dataloader, epoch)
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
                
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(f"whisper_encoder_checkpoint_epoch_{epoch + 1}.pt")
                    
            except Exception as e:
                print(f"Error during epoch {epoch}: {str(e)}")
                continue
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Usage example
if __name__ == "__main__":
    trainer = WhisperEncoderTrainer()
    trainer.train(num_epochs=10, batch_size=4, num_samples=500)
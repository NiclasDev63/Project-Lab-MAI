import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from AdaFace.inference import load_pretrained_model
import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RandomDataset(data.Dataset):
    def __init__(self, num_samples, input_size):
        self.num_samples = num_samples
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input_data = torch.randn(*self.input_size)
        return input_data

def feature_loss(features):
    """
    Placeholder LOSS
    """
    features = F.normalize(features, p=2, dim=1)
    
    similarity_matrix = torch.matmul(features, features.t())
    
    mask = torch.eye(features.size(0), device=features.device)
    loss = (torch.abs(similarity_matrix) * (1 - mask)).mean()
    
    return loss

def train_adaface():
    # Load model
    model = load_pretrained_model("ir_50")
    model = model.to(device)
    model.train()

    train_dataset = RandomDataset(
        num_samples=1000, 
        input_size=(3, 112, 112)  # 112x112 should be correct from what i could find out
    )
    
    # Training parameters just setting them here for testing
    batch_size = 64
    learning_rate = 3e-6
    num_epochs = 25
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, inputs in enumerate(progress_bar):
            last_time = time.time()
            
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            current_time = time.time()
            print(f"Forward pass took {current_time - last_time:.4f} seconds")
            last_time = current_time

            # Assuming the first element of the tuple contains the features
            features = outputs[0]  # This line might need adjustment i am just trying to get the features not the label
            
            loss = feature_loss(features)
            current_time = time.time()
            print(f"Loss computation took {current_time - last_time:.4f} seconds")
            last_time = current_time

            # Backpropagation
            loss.backward()
            print("Backward pass complete")
            current_time = time.time()
            print(f"Backward pass took {current_time - last_time:.4f} seconds")
            last_time = current_time

            optimizer.step()
            current_time = time.time()
            print(f"Optimizer step took {current_time - last_time:.4f} seconds")
            last_time = current_time
            
            # Accumulate loss
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': avg_loss})
        
        # Adjust learning rate
        scheduler.step()



    # Save final model
    torch.save(model.state_dict(), "adaface_finetuned.pth")

if __name__ == "__main__":
    train_adaface()
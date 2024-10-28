import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

# Assuming inference.py contains the model loading function
from inference import load_pretrained_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RandomDataset(data.Dataset):
    def __init__(self, num_samples, input_size):
        self.num_samples = num_samples
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Generate random image data with proper dimensions
        input_data = torch.randn(*self.input_size)
        return input_data

def feature_loss(features):
    """
    Compute loss based on visual features
    Implements a simple feature diversity loss that encourages
    features to be different from each other within the batch
    """
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.t())
    
    # We want features to be different from each other
    # So we minimize the absolute similarity between different features
    # Exclude the diagonal (self-similarity) from the loss
    mask = torch.eye(features.size(0), device=features.device)
    loss = (torch.abs(similarity_matrix) * (1 - mask)).mean()
    
    return loss

def train_adaface():
    # Load model
    model = load_pretrained_model("ir_50")
    model = model.to(device)
    model.train()

    # Create dataset and dataloader
    train_dataset = RandomDataset(
        num_samples=1000, 
        input_size=(3, 112, 112)  # AdaFace typically expects 112x112 images
    )
    
    # Training parameters
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

    # Optimizer and scheduler setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Debug first batch to understand model output
    print("Debugging model output structure:")
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_batch = sample_batch.to(device)
        outputs = model(sample_batch)
        print("Model output type:", type(outputs))
        if isinstance(outputs, tuple):
            print("Number of elements in tuple:", len(outputs))
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    print(f"Output {i} shape:", out.shape)
                else:
                    print(f"Output {i} type:", type(out))

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Assuming the first element of the tuple contains the features
            # Modify this based on the debug output above
            features = outputs[0]  # This line might need adjustment based on debug output
            
            # Calculate loss based on features
            loss = feature_loss(features)
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Avg Loss: {avg_loss:.4f}")

        # Adjust learning rate
        scheduler.step()
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
        }
        torch.save(checkpoint, f'adaface_checkpoint_epoch_{epoch+1}.pth')

    # Save final model
    torch.save(model.state_dict(), "adaface_finetuned.pth")

if __name__ == "__main__":
    train_adaface()
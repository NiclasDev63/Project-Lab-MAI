import torch

    
for i in range(1,10):
    checkpoint = torch.load(f'checkpoint_epoch_{i}.pth')
    print(f"loaded checkpoint {i} with loss {checkpoint['train_loss']}")

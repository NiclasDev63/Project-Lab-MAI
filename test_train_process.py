### Test py to simulate a full training loop with less data for testing and debugging
from crossmodal_training import create_voxceleb2_dataloader
from pathlib import Path

# Create Dataloader
dataset_dir = "datasets"
dataloader = create_voxceleb2_dataloader(Path(dataset_dir), split='test', num_workers=0)

# Test Dataloader
print(next(iter(dataloader)))
print("Done")
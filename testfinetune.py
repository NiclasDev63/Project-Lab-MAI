from collections import OrderedDict
import torch
import torchvision.transforms as transforms
from PIL import Image

from AdaFace.face_alignment import align
from AdaFace.head import AdaFace
from AdaFace.inference import to_input


# Path to the .pth file
model_path = "adaface_finetuned.pth"

checkpoint = torch.load(model_path, map_location=torch.device("cuda"))

# Adapt keys to match the expected names in the AdaFace model
adapted_checkpoint = OrderedDict()
for k, v in checkpoint.items():
    # Map keys here (modify this mapping based on your model's needs)
    if "input_layer" in k:
        new_key = k.replace("input_layer", "kernel")  # Adjust mapping logic as needed
    elif "output_layer" in k:
        new_key = k.replace("output_layer", "t")  # Adjust as per your model
    else:
        new_key = k
    adapted_checkpoint[new_key] = v

# Load the adapted checkpoint
model = AdaFace()
model.load_state_dict(adapted_checkpoint, strict=False)
model.eval()

aligned_rgb_img = align.get_aligned_face(
    "C:\\Users\\oussama\\Desktop\\WS24-25\\Practical-LAB-MAI\\Project-Lab-MAI-main\\img1.jpeg"
)
bgr_input = to_input(aligned_rgb_img)
feature, _ = model(bgr_input)

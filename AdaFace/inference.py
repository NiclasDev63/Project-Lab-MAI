import AdaFace.net as net
import torch
import os
from AdaFace.face_alignment import align
import numpy as np

import AdaFace.net as net
import torch
import os
from AdaFace.face_alignment import align
import numpy as np


adaface_models = {
    "ir_50": "AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
    "checkpoint_epoch1": "AdaFace/pretrained/checkpoint_epoch_1.pth",
    "checkpoint_epoch2": "AdaFace/pretrained/checkpoint_epoch_2.pth",
}


def load_pretrained_model(architecture="ir_50"):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model("ir_50")
    if architecture == "ir_50":
        statedict = torch.load(adaface_models[architecture])["state_dict"]
        model_statedict = {
            key[6:]: val for key, val in statedict.items() if key.startswith("model.")
        }
        model.load_state_dict(model_statedict)
    if architecture == "checkpoint_epoch1" or architecture == "checkpoint_epoch2":
        # Load the checkpoint file
        checkpoint = torch.load(
            adaface_models[architecture]
        )  # Load the full checkpoint dictionary
        statedict = checkpoint[
            "model_state_dict"
        ]  # Extract the model's state dictionary

        # Adjust keys by removing `_orig_mod.` prefix if present
        adjusted_statedict = {
            key.replace("_orig_mod.", ""): value for key, value in statedict.items()
        }

        # Load the adjusted state dictionary into the model
        model.load_state_dict(
            adjusted_statedict
        )  # strict=False allows for missing/extra keys
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


if __name__ == "__main__":

    model = load_pretrained_model("ir_50")
    feature, norm = model(torch.randn(2, 3, 112, 112))

    test_image_path = "face_alignment/test_images"
    features = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        features.append(feature)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)

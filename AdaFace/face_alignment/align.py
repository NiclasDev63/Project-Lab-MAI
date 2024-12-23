import torch.multiprocessing as mp
from AdaFace.face_alignment import mtcnn
from PIL import Image

# Global variable to hold the model
mtcnn_model = None


def initialize_model(device="cuda:0"):
    """Initialize the MTCNN model globally"""
    global mtcnn_model
    if mtcnn_model is None:
        mtcnn_model = mtcnn.MTCNN(device=device, crop_size=(112, 112))


def get_aligned_face(image_path=None, rgb_pil_image=None):
    # Initialize model if not already initialized
    if mtcnn_model is None:
        initialize_model()

    if rgb_pil_image is None:
        img = Image.open(image_path).convert("RGB")
    else:
        assert isinstance(
            rgb_pil_image, Image.Image
        ), "Face alignment module requires PIL image or path to the image"
        img = rgb_pil_image
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0]
    except Exception as e:
        print("Face detection Failed due to error.")
        print(e)
        face = None
    return face

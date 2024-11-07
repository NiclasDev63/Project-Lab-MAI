import cv2
import torch
from AdaFace.inference import load_pretrained_model
from AdaFace.face_alignment import align
from AdaFace.inference import to_input
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the pretrained AdaFace model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_pretrained_model("ir_50")  # Adjust if you have a different model name
model.to(device)
model.eval()  # Set the model to evaluation mode3
identity_root_folder = (
    "mp4"  # Root folder containing all identity folders (e.g., id08629, id08641, etc.)
)

import os
import cv2
import torch

# Parameters
temp_frames_folder = "temp_frames"  # Temporary folder to store extracted frames
target_fps = 5  # Frames per second for frame extraction
max_identities = 10  # Maximum number of identities to process
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure temp_frames_folder exists
os.makedirs(temp_frames_folder, exist_ok=True)


# Frame extraction function
def extract_frames(video_path, output_folder, target_fps):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)

    frame_count = 0
    saved_frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break

        # Save every nth frame, where n = frame_interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                output_folder, f"frame_{saved_frame_count}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    video.release()
    return output_folder


# Load image and extract features function
def load_image(image_path):
    aligned_rgb_img = align.get_aligned_face(image_path)
    bgr_tensor_input = to_input(aligned_rgb_img).to(device)
    feature, _ = model(bgr_tensor_input)
    return feature


# Function to extract identity features from frames
def extract_video_identity_features(frames_folder):
    video_identity_features = []
    for file in sorted(os.listdir(frames_folder)):
        if file.endswith(".jpg"):  # Only process .jpg files (frames)
            frame_path = os.path.join(frames_folder, file)
            print(f"Processing frame: {frame_path}")

            # Pass frame through AdaFace model
            frame_identity_features = load_image(frame_path)
            video_identity_features.append(
                frame_identity_features.squeeze()
            )  # Shape: (512)

    return torch.stack(video_identity_features)  # Shape: (T, 512)


# Main function to get identity features with frame extraction
def get_identity_features(identity_root_folder, max_identities=10):
    all_identity_features = []
    identity_count = 0

    for identity_folder in sorted(os.listdir(identity_root_folder)):
        identity_path = os.path.join(identity_root_folder, identity_folder)

        if not os.path.isdir(identity_path):
            continue

        print(f"Processing identity: {identity_folder}")
        identity_count += 1

        identity_features_per_video = []

        # Loop through each video folder within the identity folder
        for video_folder in sorted(os.listdir(identity_path)):
            video_path = os.path.join(identity_path, video_folder)

            if not os.path.isdir(video_path):
                continue

            print(f"Processing video: {video_folder}")

            # Locate the video file (assuming there's one video file per folder)
            video_file = next(
                (f for f in os.listdir(video_path) if f.endswith(".mp4")), None
            )
            if video_file is None:
                print(f"No video found in {video_folder}")
                continue

            video_file_path = os.path.join(video_path, video_file)
            frames_folder = os.path.join(
                temp_frames_folder, f"{identity_folder}_{video_folder}"
            )

            # Extract frames from the video
            extract_frames(video_file_path, frames_folder, target_fps)

            # Extract identity features from the extracted frames
            video_identity_features = extract_video_identity_features(frames_folder)
            identity_features_per_video.append(video_identity_features)

            # Clean up the frames folder after processing
            for frame_file in os.listdir(frames_folder):
                os.remove(os.path.join(frames_folder, frame_file))
            os.rmdir(frames_folder)

        # Take the first video for each identity if multiple videos exist
        if identity_features_per_video:
            all_identity_features.append(identity_features_per_video[0])

        if identity_count >= max_identities:
            break

    # Stack features from all identities
    identity_features_matrix = torch.stack(all_identity_features)  # Shape: (N, T, 512)
    return identity_features_matrix


# Call the function to get identity features for 10 identities
identity_features_matrix = get_identity_features(identity_root_folder, max_identities=2)

# Verify the shape
print("Identity features matrix shape:", identity_features_matrix.shape)

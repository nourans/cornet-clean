import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import uuid
import json
import os
from fgsm_helperfxnsOG import (
    get_all_image_paths, get_input_batch, output_prediction, extract_true_label,
    compare_labels, fgsm_attack, save_adv_image, run_fgsm_pipeline,
    extract_true_label_from_path
)

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--data_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
FLAGS, FIRE_FLAGS = parser.parse_known_args()
fnames = []
for ext in ["*.png"]:
    fnames.extend(glob.glob(os.path.join(FLAGS.data_path, "**", ext), recursive=True))


# root_dir = "val/val"
# all_images = get_all_image_paths(root_dir)

# Constants
epsilons = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
correct_pred = 0
total_images = 0
# TRIAL 2: counting correct per epsilon
correct_after_per_eps = {eps: 0 for eps in epsilons}
total_per_eps = {eps: 0 for eps in epsilons}

# Load the model once globally
model = models.alexnet(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loop through all images
for filename in fnames:
    try:
        total_images += 1

        try:
            img = Image.open(filename).convert('RGB')  # Load image
        except:
            print(f"⚠️ Unable to load image: {filename}")
            continue  # Skip the image and move on

        # Get true label
        true_label_perturbed, true_index_perturbed = extract_true_label_from_path(filename)

        input_batch = get_input_batch(device, filename)
        pred_perturbed = output_prediction(model, input_batch)

        # Compare before-attack prediction
        is_correct = compare_labels(pred_perturbed, true_index_perturbed)

        if is_correct == True: # if it passes alexnet
            correct_pred += 1

        print(f"{filename} | correct prediction after perturbation? {is_correct}: True label is {true_label_perturbed} and Predicted label is {pred_perturbed}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")



print(f"\nImages correctly predicted by AlexNet: {correct_pred}/{total_images} = {correct_pred / total_images:.2%}")
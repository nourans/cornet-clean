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
    compare_labels, fgsm_attack, save_adv_image, run_fgsm_pipeline
)

root_dir = "val/val"
all_images = get_all_image_paths(root_dir)

# constants
epsilons = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
correct_before = 0
total_images = 0
# counting correct per epsilon
correct_after_per_eps = {eps: 0 for eps in epsilons}
total_per_eps = {eps: 0 for eps in epsilons}

# load the model once globally
model = models.alexnet(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# loop through all images
for filename in all_images:
    try:
        total_images += 1

        input_batch = get_input_batch(device, filename)

        # get true label
        true_index, true_label = extract_true_label(filename)

        # get prediction before FGSM
        pred_before = output_prediction(model, input_batch)

        # compare before-attack prediction
        is_correct_before = compare_labels(pred_before, true_index)

        if is_correct_before == True: # if it passes alexnet

            correct_before += 1

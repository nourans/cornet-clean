from cornet import cornet_s
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import uuid
import json
import os
from fgsm_helperfxns import (
    get_all_image_paths, get_input_batch, output_prediction, extract_true_label,
    compare_labels, fgsm_attack, save_adv_image, run_fgsm_pipeline
)

root_dir = "val/val"
all_images = get_all_image_paths(root_dir)

# Constants
epsilons = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
correct_before = 0
total_images = 0
# TRIAL 2: counting correct per epsilon
correct_after_per_eps = {eps: 0 for eps in epsilons}
total_per_eps = {eps: 0 for eps in epsilons}

# Load the model once globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cornet_s(pretrained=True, map_location=device)  # CORnet-S model with pretrained weights
model = model.module if hasattr(model, 'module') else model
model.eval()
dummy = torch.randn(1, 3, 224, 224)
out = model(dummy)
print(out.shape)

model.to(device)

# Loop through all images
for filename in all_images:
    try:
        total_images += 1

        input_batch = get_input_batch(device, filename)

        # Get true label
        true_index, true_label = extract_true_label(filename)
        print(f"✅ True label: {true_label}, index: {true_index}")

        # Get prediction before FGSM (this returns a string label)
        pred_before = output_prediction(model, input_batch)

        print(f"📊 Model predicted: {pred_before}, True index: {true_index}")

        # Compare string vs string (use true_label, not true_index)
        is_correct_before = compare_labels(pred_before, true_label)
        if is_correct_before:
            correct_before += 1

            for eps in epsilons:
                # … run_fgsm_pipeline returns an int index …
                # pred_after = run_fgsm_pipeline(model, device, filename, eps)
                # added jun 10: start
                # Load AlexNet (once globally above or here if dynamic)
                alexnet = models.alexnet(pretrained=True).to(device)
                alexnet.eval()

                # Use CORnet to generate perturbed image
                pred_after_cornet, perturbed_image = run_fgsm_pipeline(model, device, filename, eps)
                save_adv_image(
                    perturbed_image, eps, true_label, true_index, pred_before, pred_after_cornet,
                    output_dir=f"adv_CORoutputs5/adv_CORoutputs5_eps{eps}"
                )
                total_per_eps[eps] += 1
                # added jun 10: end

                # Compare indices (both ints)
                is_correct_after = compare_labels(pred_after_cornet, true_index)
                if is_correct_after:
                    correct_after_per_eps[eps] += 1
                
                # Add debugging output
                print(f"  Epsilon {eps}: Predicted {pred_after_cornet}, True {true_index}, Correct: {is_correct_after}")
                # print(f"Correct for epsilon = {eps} is {correct_after}") # TRIAL 2: delete this

                # save_adv_image(
                #     perturbed_image, eps, true_label, true_index, pred_before, pred_after_cornet,
                #     output_dir=f"adv_outputs13/adv_outputs13_eps{eps}"
                # )
                # total_per_eps[eps] += 1
        print(f"---------------------------------------------------------{filename} ends---------------------------------------------------------")
    except Exception as e:
        print(f"Error processing {filename}: {e}")


# print(f"\nCorrect before FGSM: {correct_before}/{total_images} = {correct_before / total_images:.2%}")
# for eps in epsilons:
#     acc = correct_after_per_eps[eps] / total_per_eps[eps]
#     print(f"Epsilon {eps}: Accuracy after FGSM = {correct_after_per_eps[eps]}/{total_per_eps[eps]} = {acc:.2%}")

print(f"\nCorrect before FGSM: {correct_before}/{total_images} = {correct_before / total_images:.2%}")
for eps in epsilons:
    if total_per_eps[eps] == 0:
        print(f"Epsilon {eps}: ⚠️ No adversarial images saved or processed.")
    else:
        # might wanna use current_count instead of total_per_eps[eps] for accuracy calculations.
        # it works now, but may cause problems in the future
        acc = correct_after_per_eps[eps] / total_per_eps[eps]
        print(f"Epsilon {eps}: Accuracy after FGSM = {correct_after_per_eps[eps]}/{total_per_eps[eps]} = {acc:.2%}")

        




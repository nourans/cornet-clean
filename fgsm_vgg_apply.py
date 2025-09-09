import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
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
# TRIAL 2: counting correct per epsilon
correct_after_per_eps = {eps: 0 for eps in epsilons}
total_per_eps = {eps: 0 for eps in epsilons}

# load the model once globally
model = models.vgg16(pretrained=True)
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

            for eps in epsilons:
                pred_after = run_fgsm_pipeline(model, device, filename, eps) # run fgsm attack
                # TRIAL 3: try commenting the line below out to see if count can be fixed
                # total_per_eps[eps] += 1 # TRIAL 2: counting here how many runs are we getting with each eps

                """Save perturbed image (need input tensor againbecause the previous input_batch may have been 
                modified by the FGSM attack — so if we reused it for the next epsilon, the attack wouldn't be 
                applied to the original clean image but to an already-perturbed one)"""
                # so re-generate it here:
                input_batch = get_input_batch(device, filename)
                input_batch.requires_grad = True
                output = model(input_batch)
                loss = F.nll_loss(F.log_softmax(output, dim=1), torch.tensor([true_index]).to(device))
                model.zero_grad()
                loss.backward()
                data_grad = input_batch.grad.data
                perturbed_image = fgsm_attack(input_batch, eps, data_grad)
                try:
                    save_adv_image(perturbed_image, eps, true_label, true_index, pred_before, pred_after, output_dir=f"adv_VGGoutputs3/adv_VGGoutputs3_eps{eps}")
                    total_per_eps[eps] += 1 # TRIAL 3: counting here how many runs are we getting with each eps
                except Exception as e:
                    print(f"❌ Failed to save image for {filename} at eps={eps}: {e}")
                # log or compare accuracy
                is_correct_after = compare_labels(pred_after, true_index)
                print(f"{filename} | eps={eps} | correct before? {is_correct_before} | correct after? {is_correct_after}")
                if is_correct_after == True:
                    correct_after_per_eps[eps] += 1
                # print(f"Correct for epsilon = {eps} is {correct_after}") # TRIAL 2: delete this
        print(f"---------------------------------------------------------{filename} ends---------------------------------------------------------")
    except Exception as e:
        print(f"Error processing {filename}: {e}")


print(f"\nCorrect before FGSM: {correct_before}/{total_images} = {correct_before / total_images:.2%}")
for eps in epsilons:
    if total_per_eps[eps] == 0:
        print(f"Epsilon {eps}: ⚠️ No adversarial images saved or processed.")
    else:
        # might wanna use current_count instead of total_per_eps[eps] for accuracy calculations.
        # it works now, but may cause problems in the future
        acc = correct_after_per_eps[eps] / total_per_eps[eps]
        print(f"Epsilon {eps}: Accuracy after FGSM = {correct_after_per_eps[eps]}/{total_per_eps[eps]} = {acc:.2%}")


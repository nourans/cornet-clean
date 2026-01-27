### FGSM + AlexNet Combined Notebook

# --- Section 1: Imports and Setup ---
import os, glob, json, time, re
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import tqdm

# Helper functions (adapted from fgsm_helperfxnsOG)
def get_all_image_paths(root_dir):
    image_extensions = ["*.JPEG", "*.jpeg", "*.JPG", "*.jpg", "*.png"]
    paths = []
    for ext in image_extensions:
        paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return paths

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_input_batch(device, filename):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])
    image = Image.open(filename).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

with open("imagenet_class_index.json") as f:
    imagenet_classes = json.load(f)
imagenet_classes = {int(k): v[1] for k,v in imagenet_classes.items()}

def output_prediction(model, input_batch):
    with torch.no_grad():
        output = model(input_batch)
    _, pred = torch.max(output, 1)
    return pred.item()

def extract_true_label(filepath):
    # assumes val/val/class_xxx/... structure OR adversarial filename pattern
    parent = os.path.basename(os.path.dirname(filepath))
    if "_" in parent:
        return None, parent.split("_",1)[1]
    stem = os.path.splitext(os.path.basename(filepath))[0]
    return None, stem.split("_")[-1]

def compare_labels(pred_index, true_index):
    return pred_index == true_index

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)


def save_adv_image(tensor, eps, true_label, true_index, pred_before, pred_after, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"adv_{true_label}_eps{eps}_pred{pred_after}.png"
    save_path = os.path.join(output_dir, filename)
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
        std=[1/s for s in [0.229,0.224,0.225]]
    )
    img = inv_normalize(tensor.squeeze().cpu()).clamp(0,1)
    img = transforms.ToPILImage()(img)
    img.save(save_path)


# --- Section 2: FGSM Experiment Pipeline ---

def run_fgsm_experiment(root_dir, epsilons, output_root="adv_ALEX_outputs"):
    all_images = get_all_image_paths(root_dir)
    
    correct_before = 0
    total_images = 0
    correct_after_per_eps = {eps:0 for eps in epsilons}
    total_per_eps = {eps:0 for eps in epsilons}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.alexnet(pretrained=True).to(device)
    model.eval()

    for filename in tqdm.tqdm(all_images):
        try:
            total_images += 1
            input_batch = get_input_batch(device, filename)

            _, true_label = extract_true_label(filename)

            pred_before = output_prediction(model, input_batch)

            # compare before attack
            if imagenet_classes.get(pred_before,"?") == true_label:
                correct_before += 1

                for eps in epsilons:
                    input_batch = get_input_batch(device, filename)
                    input_batch.requires_grad = True
                    output = model(input_batch)
                    target = torch.tensor([pred_before]).to(device)
                    loss = F.nll_loss(F.log_softmax(output, dim=1), target)
                    model.zero_grad()
                    loss.backward()

                    data_grad = input_batch.grad.data
                    perturbed_image = fgsm_attack(input_batch, eps, data_grad)

                    output = model(perturbed_image)
                    _, pred_after = torch.max(output, 1)

                    save_adv_image(perturbed_image, eps, true_label, None, pred_before, pred_after.item(), os.path.join(output_root,f"eps{eps}"))

                    total_per_eps[eps]+=1
                    if imagenet_classes.get(pred_after.item(),"?") == true_label:
                        correct_after_per_eps[eps]+=1
        except Exception as e:
            print(f"Error {filename}: {e}")

    print(f"Correct before FGSM: {correct_before}/{total_images} = {correct_before/total_images:.2%}")
    for eps in epsilons:
        if total_per_eps[eps]>0:
            acc = correct_after_per_eps[eps]/total_per_eps[eps]
            print(f"Epsilon {eps}: {correct_after_per_eps[eps]}/{total_per_eps[eps]} = {acc:.2%}")
        else:
            print(f"Epsilon {eps}: No samples.")


# --- Section 3: AlexNet Testing Pipeline ---

def test_alexnet_on_folder(data_path, imsize=224):
    model = models.alexnet(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((imsize,imsize)),
        transforms.ToTensor(),
        normalize
    ])

    fnames=[]
    for ext in ["*.JPEG","*.jpeg","*.JPG","*.jpg","*.png"]:
        fnames.extend(glob.glob(os.path.join(data_path,"**",ext), recursive=True))
    if len(fnames)==0:
        raise FileNotFoundError(f"No files in {data_path}")

    count=0
    for f in tqdm.tqdm(fnames):
        try:
            im = Image.open(f).convert('RGB')
        except:
            continue
        im = transform(im).unsqueeze(0)
        outputs = model(im)
        _, pred_idx = torch.max(outputs,1)
        pred_label = imagenet_classes.get(pred_idx.item(),"Unknown")

        true_label = extract_true_label(f)[1]

        if true_label == pred_label:
            count+=1

    accuracy = (count/len(fnames))*100
    print(f"Accuracy: {count}/{len(fnames)} = {accuracy:.2f}%")


# run_fgsm_experiment("val/val", [0.01,0.05,0.1])
# test_alexnet_on_folder("adv_ALEX_outputs/eps0.05")

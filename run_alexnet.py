# to run this: python3 run_alexnet.py test --data_path "./adv_CORoutputs4/adv_CORoutputs4_eps0.01" --model S > "alexnet_pred_cor_test1_all_new_eps0.01.txt"
import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
import torchvision.models as models

from PIL import Image
Image.warnings.simplefilter('ignore')

import re
import json
import fnmatch

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--data_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S'], default='Z',
                    help='which model to train')
parser.add_argument('--times', default=5, type=int,
                    help='number of time steps to run the model (only R model)')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')


FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)


def get_model(pretrained=False):
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    
    # load ALEXNET
    model = models.alexnet(pretrained=True)

    if FLAGS.ngpus == 0:
        model = model.module if hasattr(model, 'module') else model
    if FLAGS.ngpus > 0:
        model = model.cuda()
    return model


def extract_true_label_from_path(image_path: str) -> str:
    import re, os
    basename = os.path.basename(image_path)
    # 1. adversarial pattern
    m = re.search(r'^[^_]+_([A-Za-z_]+?)\d+_init', basename)
    if m:
        return m.group(1)
    # 2. standard imagenet val folder
    parent = os.path.basename(os.path.dirname(image_path))
    if '_' in parent:
        label_part = parent.split('_', 1)[1]
        return label_part
    # 3. fallback from filename
    stem = os.path.splitext(basename)[0]
    pieces = stem.split('_')
    if pieces:
        return pieces[-1]
    return "Unknown"


def test(layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    Tests the AlexNet model on a set of adversarial images and prints the predicted labels.

    Args:
        - layer (not used here, but in original feature extraction)
        - sublayer (not used here)
        - time_step (not used here)
        - imsize (default 224): The size to resize input images to
    """
    # load the model with pretrained weights
    model = get_model(pretrained=True)

    # define preprocessing transformations for input images
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    # set model to evaluation mode (disables dropout, batchnorm updates)
    model.eval()

    # load ImageNet class labels from a file (fixes "Unknown" issue)
    imagenet_classes = {i: label.strip() for i, label in enumerate(open("imagenet_classes.txt").readlines())}
    
    
    # get all image file paths from the dataset directory (recursively search subdirectories)
    fnames = []
    for ext in ["*.JPEG", "*.jpeg", "*.JPG", "*.jpg", "*.png"]:
        fnames.extend(glob.glob(os.path.join(FLAGS.data_path, "**", ext), recursive=True))

    # if no images are found, raise eror
    if len(fnames) == 0:
        raise FileNotFoundError(f'No files found in {FLAGS.data_path}')

    print(f"{FLAGS.data_path}\n\n")
    count = 0
    # Iterate over images, process them, and pass through the model
    with torch.no_grad():
        for image_path in tqdm.tqdm(fnames, desc="Testing Images"):
            try:
                im = Image.open(image_path).convert('RGB')  # Load image
            except:
                print(f"⚠️ Unable to load image: {image_path}")
                continue  # Skip the image and move on

            im = transform(im)  # Apply transformations
            im = im.unsqueeze(0)  # Add batch dimension

            # Run the image through the model
            outputs = model(im)  # Forward pass
            _, predicted_index = torch.max(outputs, 1)  # Get highest logit index
            #CHECKPOINT:
            print(f"\n🔍 AlexNet Predicted Index: {predicted_index.item()}")

            ##################################################################################
            with open("imagenet_class_index.json") as f:
                imagenet_classes = json.load(f)
            imagenet_classes = {int(k): v[1] for k, v in imagenet_classes.items()}
            # if predicted_index.item() in imagenet_classes:
            #     predicted_label = imagenet_classes[predicted_index.item()]
            ################################################################################

            # convert predicted index to human-readable label
            predicted_label = imagenet_classes.get(predicted_index.item(), "Unknown")
            # CHECKPOINT:
            print(f"🧐 Predicted Class Name: {predicted_label}")

            # extract true label (supports adversarial names and val/val folders)
            basename = os.path.basename(image_path)
            true_label = extract_true_label_from_path(image_path)

            # compare with predicted label
            if true_label == predicted_label:
                print("✅ MATCH!")
                count += 1
            else:
                print("❌ NOPE")

            # print results in a clear format
            print(f"{basename} → True: {true_label} | AlexNet: {predicted_label}")
            #print(f"{os.path.basename(image_path)} → True: {true_label} | AlexNet: {predicted_label}")
    
    length = len(fnames)
    accuracy = (count/length)*100
    print(f"Testing Complete. Results saved to alexnet_exp1_2019_results.csv\n Correct count is {count}\n Total images is {length}\n Accuracy is {accuracy:.2f}")


class ImageNetTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         FLAGS.lr,
                                         momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.step_size)
        self.loss = nn.CrossEntropyLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=True,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        if FLAGS.ngpus > 0:
            target = target.cuda(non_blocking=True)
        output = self.model(inp)

        record = {}
        loss = self.loss(output, target)
        record['loss'] = loss.item()
        record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        record['top1'] /= len(output)
        record['top5'] /= len(output)
        record['learning_rate'] = self.lr.get_lr()[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record


class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'val_in_folders'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                if FLAGS.ngpus > 0:
                    target = target.cuda(non_blocking=True)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire({
        'test': test
    }) 
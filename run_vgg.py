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

import cornet

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
    
    # Load VGG model instead of CORnet
    if FLAGS.model.lower() == 's':
        model = models.vgg16(pretrained=True)
    elif FLAGS.model.lower() == 'r':
        model = models.vgg19(pretrained=True)
    elif FLAGS.model.lower() == 'z':
        model = models.vgg11(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)  # default to VGG16

    if FLAGS.ngpus == 0:
        model = model.module if hasattr(model, 'module') else model
    if FLAGS.ngpus > 0:
        model = model.cuda()
    return model


def train(restore_path=None,  # useful when you want to restart training
          save_train_epochs=.1,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=5,  # how often save model weigths
          save_model_secs=60 * 10  # how often save model (in sec)
          ):

    model = get_model()
    trainer = ImageNetTrain(model)
    validator = ImageNetVal(model)

    start_epoch = 0
    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])

    records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in tqdm.trange(0, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan
        for step, data in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()
                    trainer.model.train()

            if FLAGS.output_path is not None:
                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results.pkl'), 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           f'epoch_{epoch:02d}.pth.tar'))

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record

            data_load_start = time.time()


def extract_true_label(basename: str) -> str:
    import re
    #  EITHER capture everything between the first "_" and the digits before "_init"
    m = re.search(r'^[^_]+_([A-Za-z_]+?)\d+_init', basename)
    if m:
        return m.group(1)
        # .replace('_', ' ')  # e.g. 'black_swan' -> 'black swan'
    # ORRR join tokens until we hit 'init...'; then strip trailing digits
    toks = basename.split('_')[1:]  # skip epsilon at index 0
    acc = []
    for t in toks:
        if t.startswith('init'):
            break
        acc.append(t)
    label_with_idx = '_'.join(acc)                 # eg black_swan100
    label = re.sub(r'\d+$', '', label_with_idx)    # becomes black_swan
    return label
    # .replace('_', ' ')


def extract_true_label_from_path(image_path: str) -> str:
    import re, os
    basename = os.path.basename(image_path)
    # adversarial pattern: <eps>_<label_with_idx>_init... (e.g., 0.005_bee_eater92_init...)
    m = re.search(r'^[^_]+_([A-Za-z_]+?)\d+_init', basename)
    if m:
        return m.group(1)
    # standard imagenet val folder: .../nxxxxx_label/filename
    parent = os.path.basename(os.path.dirname(image_path))
    if '_' in parent:
        label_part = parent.split('_', 1)[1]  # after first underscore
        return label_part
    # use last underscore token in filename (before extension)
    stem = os.path.splitext(basename)[0]
    pieces = stem.split('_')
    if pieces:
        return pieces[-1]
    return "Unknown"

def test(layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    Tests the CORnet model on a set of images and prints the predicted labels.

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

    # load imagenet class labels from a file (fixes "Unknown" issue)
    imagenet_classes = {i: label.strip() for i, label in enumerate(open("imagenet_classes.txt").readlines())}
    
    
    # get all image file paths from the dataset directory
    fnames = []
    for ext in ["*.JPEG", "*.jpeg", "*.JPG", "*.jpg", "*.png"]:
        fnames.extend(glob.glob(os.path.join(FLAGS.data_path, "**", ext), recursive=True))
    # fnames = sorted(glob.glob(os.path.join(FLAGS.data_path, '**', '*.png', '*.JPEG', ), recursive=True))

    # if no images are found, raise an error
    if len(fnames) == 0:
        raise FileNotFoundError(f'No files found in {FLAGS.data_path}')

    print(f"{FLAGS.data_path}\n\n")
    count = 0
    # iterate over images, process them, and pass through the model
    with torch.no_grad():
        for image_path in tqdm.tqdm(fnames, desc="Testing Images"):
            try:
                im = Image.open(image_path).convert('RGB')  # Load image
            except:
                print(f"⚠️ Unable to load image: {image_path}")
                continue  # Skip the image and move on

            im = transform(im)  # apply transformations
            im = im.unsqueeze(0)  # ad batch dimension

            # run the image through the model
            outputs = model(im)  # Forward pass
            _, predicted_index = torch.max(outputs, 1)  # Get highest logit index
            #CHECKPOINT:
            print(f"\n🔍 VGG Predicted Index: {predicted_index.item()}")
            #################################################################################
            with open("imagenet_class_index.json") as f:
                imagenet_classes = json.load(f)
            imagenet_classes = {int(k): v[1] for k, v in imagenet_classes.items()}
            # if predicted_index.item() in imagenet_classes:
            #     predicted_label = imagenet_classes[predicted_index.item()]
            #####

            # convert predicted index to human-readable label
            predicted_label = imagenet_classes.get(predicted_index.item(), "Unknown")
            # CHECKPOINT:
            print(f"🧐 Predicted Class Name: {predicted_label}")

            # extract true label from filename (assume format: "0.005_knot616_init616_adv616_23d77fcb.png")
            basename = os.path.basename(image_path)
            true_label = extract_true_label_from_path(image_path)



            # compare with predicted label
            if true_label == predicted_label:
                print("✅ MATCH!")
                count += 1
            else:
                print("❌ NOPE")

            # print results in a clear format
            print(f"{basename} → True: {true_label} | VGG: {predicted_label}")
            #print(f"{os.path.basename(image_path)} → True: {true_label} | CORnet: {predicted_label}")
    
    # use the actual number of image files found instead of counting files in root directory
    length = len(fnames)
    accuracy = (count/length)*100
    print(f"Testing Complete. Results saved to cornet_exp1_2019_results.csv\n Correct count is {count}\n Total images is {length}\n Accuracy is {accuracy:.2f}")


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


# if __name__ == '__main__':
#     fire.Fire(command=FIRE_FLAGS)

if __name__ == '__main__':
    fire.Fire({
        'test': test,
        'train': train
    })

# metamazium/train_model/train_snail_mini.py
import os
import json
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
import glob
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

# Import the SNAIL model from snail_c.py
from metamazium.snail_performer.snail_c import SNAILFewShot

#########################################
# 1. MiniImagenet Dataset
#########################################
class MiniImagenetDataset(Dataset):
    """
    A custom dataset for miniImagenet.
    Expects a JSON file with keys:
      - "label_names": list of class names.
      - "image_names": list of image file paths (relative to data_dir/images).
      - "image_labels": list of integer labels.
    
    This version strips the "filelists/miniImagenet/" prefix from the image paths,
    since your actual images are stored directly under data_dir/images/<class_name>/.
    """
    def __init__(self, json_file, data_dir, transform=None):
        super(MiniImagenetDataset, self).__init__()
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.label_names = data["label_names"]
        self.image_paths = data["image_names"]
        # Remove the unwanted prefix from each image path.
        self.image_paths = [p.replace("filelists/miniImagenet/", "") for p in self.image_paths]
        self.image_labels = data["image_labels"]
        # Remap labels to 0...num_classes-1
        unique_labels = sorted(list(set(self.image_labels)))
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        self.image_labels = [self.label_map[l] for l in self.image_labels]
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Construct full path (assumes images are in data_dir/images/)
        img_path = os.path.join(self.data_dir, "images", self.image_paths[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.image_labels[index]
        return img, label

#########################################
# 2. Episode Sampling Utilities
#########################################
def label_idx(dataset):
    """
    Builds a dictionary mapping each label to a list of indices.
    """
    label_index = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_index.setdefault(label, []).append(idx)
    return label_index

def ep_sampler(label_index, dataset, n_way=5, k_shot=1, device="cpu"):
    """
    Samples a single N-way, K-shot episode.
    Returns:
      support_images: list of support image tensors
      support_labels: list of one-hot vectors (dimension n_way) for support images
      query_image: query image tensor
      query_target: integer label (0 to n_way-1) for the query
      query_label_dummy: dummy one-hot vector (all zeros) for the query.
    """
    available_classes = list(label_index.keys())
    chosen_classes = random.sample(available_classes, n_way)
    support_imgs = []
    support_lbls = []
    support_indices = []
    for epi_label, cls in enumerate(chosen_classes):
        indices = label_index[cls]
        picks = random.sample(indices, k_shot)
        for idx in picks:
            img, _ = dataset[idx]
            support_imgs.append(img)
            support_indices.append(idx)
            one_hot = torch.zeros(n_way, dtype=torch.float)
            one_hot[epi_label] = 1.0
            support_lbls.append(one_hot)
    # For the query, choose one of the chosen classes
    query_class = random.choice(chosen_classes)
    query_epi_label = chosen_classes.index(query_class)
    available_indices = set(label_index[query_class])
    used_indices = set(idx for idx in support_indices if idx in label_index[query_class])
    candidate_indices = list(available_indices - used_indices)
    if candidate_indices:
        query_idx = random.choice(candidate_indices)
    else:
        query_idx = random.choice(list(available_indices))
    query_img, _ = dataset[query_idx]
    query_label_dummy = torch.zeros(n_way, dtype=torch.float)
    return support_imgs, support_lbls, query_img, query_epi_label, query_label_dummy

def batch_for_few_shot(label_index, dataset, n_way=5, k_shot=1, batch_size=16, device="cpu"):
    """
    Samples a batch of episodes.
    Returns:
      batch_images: Tensor of shape (batch_size*T, 3, 84, 84)
      batch_labels: Tensor of shape (batch_size*T, n_way)
      query_targets: Tensor of shape (batch_size,) with integer labels for the query images.
    Where T = n_way * k_shot + 1.
    """
    T = n_way * k_shot + 1
    batch_images = []
    batch_labels = []
    query_targets = []
    for _ in range(batch_size):
        supp_imgs, supp_lbls, query_img, query_target, query_label_dummy = ep_sampler(
            label_index, dataset, n_way=n_way, k_shot=k_shot, device=device)
        batch_images.extend(supp_imgs)
        batch_labels.extend(supp_lbls)
        batch_images.append(query_img)
        batch_labels.append(query_label_dummy)
        query_targets.append(query_target)
    batch_images = torch.stack(batch_images, dim=0).to(device)
    batch_labels = torch.stack(batch_labels, dim=0).to(device)
    query_targets = torch.tensor(query_targets, dtype=torch.long, device=device)
    return batch_images, batch_labels, query_targets

#########################################
# 3. Training Routine for miniImagenet with Checkpointing
#########################################
def train_snail(epochs=50, iterations=100, n_way=5, k_shot=1, batch_size=16,
                lr=1e-4, device="cpu", data_dir="miniimagenet"):
    # Define transforms for 84x84 images.
    transform = T.Compose([
        T.Resize((84,84)),
        T.ToTensor()
    ])
    # Load JSON files (assume they are in data_dir)
    train_json = os.path.join(data_dir, "base.json")
    val_json = os.path.join(data_dir, "val.json")
    test_json = os.path.join(data_dir, "novel.json")
    
    train_dataset = MiniImagenetDataset(train_json, data_dir, transform=transform)
    val_dataset   = MiniImagenetDataset(val_json, data_dir, transform=transform)
    test_dataset  = MiniImagenetDataset(test_json, data_dir, transform=transform)
    
    # Build label indices for episode sampling.
    train_label_index = label_idx(train_dataset)
    val_label_index = label_idx(val_dataset)
    test_label_index = label_idx(test_dataset)
    
    # Instantiate SNAILFewShot with the miniimagenet encoder.
    model = SNAILFewShot(N=n_way, K=k_shot, task='miniimagenet', use_cuda=(device=="cuda"))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Create checkpoint folder if it doesn't exist.
    ckpt_dir = "min_checkpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Check for existing checkpoints and resume if available.
    start_epoch = 0
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if checkpoint_files:
        # Function to extract epoch number from filename.
        def get_epoch(fname):
            match = re.search(r'epoch_(\d+).pth', fname)
            return int(match.group(1)) if match else -1
        checkpoint_files.sort(key=get_epoch)
        last_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch} using checkpoint: {last_checkpoint}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    print("Starting training on miniImagenet...")
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        with tqdm(total=iterations, desc=f"Epoch [{epoch+1}/{epochs}]", unit="iter") as pbar:
            for it in range(iterations):
                optimizer.zero_grad()
                images, labels, query_targets = batch_for_few_shot(train_label_index, train_dataset,
                                                                   n_way=n_way, k_shot=k_shot,
                                                                   batch_size=batch_size, device=device)
                outputs = model(images, labels)  # (batch_size, n_way)
                loss = loss_fn(outputs, query_targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                preds = outputs.argmax(dim=1)
                acc = (preds == query_targets).float().mean().item()
                epoch_acc += acc

                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.2f}%")
                pbar.update(1)
        avg_loss = epoch_loss / iterations
        avg_acc = (epoch_acc / iterations) * 100
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}  Query Acc: {avg_acc:.2f}%")
        
        # Evaluate on validation episodes after each epoch.
        model.eval()
        val_episodes = 100
        correct = 0
        with torch.no_grad():
            for _ in range(val_episodes):
                images, labels, query_targets = batch_for_few_shot(val_label_index, val_dataset,
                                                                   n_way=n_way, k_shot=k_shot,
                                                                   batch_size=1, device=device)
                outputs = model(images, labels)
                pred = outputs.argmax(dim=1)
                if pred.item() == query_targets.item():
                    correct += 1
        val_acc = 100.0 * correct / val_episodes
        print(f"[Validation] {n_way}-way, {k_shot}-shot Accuracy: {val_acc:.2f}%")
        
        # Save checkpoint after each epoch.
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
    
    # Final evaluation on test episodes.
    model.eval()
    test_episodes = 100
    correct = 0
    with torch.no_grad():
        for _ in range(test_episodes):
            images, labels, query_targets = batch_for_few_shot(test_label_index, test_dataset,
                                                               n_way=n_way, k_shot=k_shot,
                                                               batch_size=1, device=device)
            outputs = model(images, labels)
            pred = outputs.argmax(dim=1)
            if pred.item() == query_targets.item():
                correct += 1
    test_acc = 100.0 * correct / test_episodes
    print(f"\n[Test] {n_way}-way, {k_shot}-shot Accuracy over {test_episodes} episodes: {test_acc:.2f}%")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train SNAIL Few-Shot on miniImagenet")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per epoch")
    parser.add_argument("--n_way", type=int, default=5, help="N-way classification")
    parser.add_argument("--k_shot", type=int, default=1, help="K-shot support examples")
    parser.add_argument("--batch_size", type=int, default=16, help="Episodes per batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--data_dir", type=str, default="miniimagenet", help="Path to miniImagenet folder")
    args, unknown = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    model = train_snail(epochs=args.epochs,
                        iterations=args.iterations,
                        n_way=args.n_way,
                        k_shot=args.k_shot,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        device=args.device,
                        data_dir=args.data_dir)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "snail_miniimagenet.pth"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()

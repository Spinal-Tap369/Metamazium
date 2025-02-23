# metamazium/train_model/train_lstm_sl.py

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from metamazium.lstm_trpo.lstm_sl import LSTMSLFewShot

# ---------------------------
# Episode and Data Utilities
# ---------------------------
def label_idx(dataset):
    """
    Builds a dictionary mapping each label to a list of indices.
    """
    label_index = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_index.setdefault(label, []).append(idx)
    return label_index

def ep_sampler(label_index, dataset, n_way=5, k_shot=1, device="cuda"):
    """
    Samples a single N-way, K-shot episode.
    
    Returns:
      support_images: list of support image tensors
      support_labels: list of one-hot vectors (dimension n_way) for support images
      query_image: query image tensor
      query_target: integer label (0 to n_way-1) for the query
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

def batch_for_few_shot(label_index, dataset, n_way=5, k_shot=1, batch_size=16, device="cuda"):
    """
    Samples a batch of episodes.
    
    Returns:
      batch_images: Tensor of shape (batch_size*T, 1, 28, 28)
      batch_labels: Tensor of shape (batch_size*T, n_way)
      query_targets: Tensor of shape (batch_size,) with integer labels for the query images.
    
    Here T = n_way * k_shot + 1 (support + query)
    """
    T = n_way * k_shot + 1
    batch_images = []
    batch_labels = []
    query_targets = []
    for _ in range(batch_size):
        supp_imgs, supp_lbls, query_img, query_target, query_label_dummy = ep_sampler(label_index, dataset, n_way, k_shot, device)
        batch_images.extend(supp_imgs)
        batch_labels.extend(supp_lbls)
        batch_images.append(query_img)
        batch_labels.append(query_label_dummy)
        query_targets.append(query_target)
    batch_images = torch.stack(batch_images, dim=0).to(device)
    batch_labels = torch.stack(batch_labels, dim=0).to(device)
    query_targets = torch.tensor(query_targets, dtype=torch.long, device=device)
    return batch_images, batch_labels, query_targets

# ---------------------------
# Training Routine
# ---------------------------
def train_lstm_sl(epochs=50, iterations=100, n_way=5, k_shot=1, batch_size=16, lr=1e-4, device="cuda"):
    transform = T.Compose([T.Resize((28,28)), T.ToTensor()])
    train_dataset = datasets.Omniglot(root="omniglot_data", background=True, download=True, transform=transform)
    test_dataset  = datasets.Omniglot(root="omniglot_data", background=False, download=True, transform=transform)
    
    train_label_index = label_idx(train_dataset)
    test_label_index = label_idx(test_dataset)
    
    model = LSTMSLFewShot(N=n_way, K=k_shot, task='omniglot')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        with tqdm(total=iterations, desc=f"Epoch [{epoch+1}/{epochs}]", unit="iter") as pbar:
            for it in range(iterations):
                optimizer.zero_grad()
                images, labels, query_targets = batch_for_few_shot(train_label_index, train_dataset,
                                                                   n_way=n_way, k_shot=k_shot,
                                                                   batch_size=batch_size, device=device)
                outputs = model(images, labels)
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
    print(f"\n[Test] {n_way}-way, {k_shot}-shot Accuracy over {test_episodes} episodes: {100.0 * correct / test_episodes:.2f}%")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM Few-Shot on Omniglot")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--iterations", type=int, default=10000, help="Iterations per epoch")
    parser.add_argument("--n_way", type=int, default=5, help="N-way classification")
    parser.add_argument("--k_shot", type=int, default=1, help="K-shot support examples")
    parser.add_argument("--batch_size", type=int, default=32, help="Episodes per batch")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cpu' or 'cuda'")
    args, unknown = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    model = train_lstm_sl(epochs=args.epochs,
                          iterations=args.iterations,
                          n_way=args.n_way,
                          k_shot=args.k_shot,
                          batch_size=args.batch_size,
                          lr=args.lr,
                          device=args.device)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "lstm_sl_omniglot.pth"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()

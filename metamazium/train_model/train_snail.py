# metamazium/train_model/train_snail.py

import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import TensorDataset
from torchvision.datasets import Omniglot
from tqdm import tqdm
import argparse

# Import the SNAILFewShot model from our snail_c.py file.
from metamazium.snail_performer.snail_c import SNAILFewShot

##############################################
# Dataset loading (using precomputed omniglot90_{train,test}.pt)
##############################################

def build_rotated_omniglot_pt(root="omniglot", download=True, background=True):
    os.makedirs(root, exist_ok=True)
    subset_str = "train" if background else "test"
    save_path = os.path.join(root, f"omniglot90_{subset_str}.pt")
    if os.path.exists(save_path):
        data = torch.load(save_path)
        return TensorDataset(data["images"], data["labels"])
    print(f"Creating rotated Omniglot {subset_str} dataset...")
    base_ds = Omniglot(root=root, background=background, download=download, transform=None)
    transform = T.Compose([
        T.Grayscale(),
        T.Resize((28, 28)),
        T.ToTensor()
    ])
    images_list = []
    labels_list = []
    for i in tqdm(range(len(base_ds)), desc=f"Precompute {subset_str}", unit="img"):
        pil_img, base_label = base_ds[i]
        for rot in range(4):
            if rot == 0:
                img_t = transform(pil_img)
            else:
                rotated = T.functional.rotate(pil_img, 90 * rot)
                img_t = transform(rotated)
            new_label = base_label * 4 + rot
            images_list.append(img_t)
            labels_list.append(new_label)
    images = torch.stack(images_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)
    torch.save({"images": images, "labels": labels}, save_path)
    print(f"Saved precomputed Omniglot90 {subset_str} => {save_path}")
    return TensorDataset(images, labels)

def load_omniglot_train_test(root="omniglot", download=True):
    train_ds = build_rotated_omniglot_pt(root=root, download=download, background=True)
    test_ds  = build_rotated_omniglot_pt(root=root, download=download, background=False)
    return train_ds, test_ds

##############################################
# Episode Sampling (per episode)
##############################################
def sample_episode(dataset, n_way=5, k_shot=1, device="cuda"):
    """
    Sample an N-way, K-shot episode as described in the paper:
      - Randomly select N distinct classes.
      - For each class, sample K support images.
      - Then, randomly pick one of these N classes and sample one additional image as the query.
      
    Returns:
      support_images: Tensor of shape (N*K, 1, 28, 28)
      support_labels: Tensor of shape (N*K, N) (one-hot)
      query_image:    Tensor of shape (1, 1, 28, 28)
      query_label:    Tensor of shape (1,) with integer in [0, N-1]
    """
    images, labels = dataset.tensors
    images = images.to(device)
    labels = labels.to(device)
    # Build a dictionary mapping class label to indices.
    class_to_indices = {}
    for i, lab in enumerate(labels):
        c = int(lab.item())
        class_to_indices.setdefault(c, []).append(i)
    all_classes = list(class_to_indices.keys())
    chosen_classes = random.sample(all_classes, n_way)
    sup_imgs, sup_lbls = [], []
    for epi_label, c in enumerate(chosen_classes):
        idx_pool = class_to_indices[c]
        picks = random.sample(idx_pool, k_shot)
        for si in picks:
            sup_imgs.append(images[si])
            one_hot = torch.zeros(n_way, device=device)
            one_hot[epi_label] = 1.0
            sup_lbls.append(one_hot)
    # For the query, randomly pick one of the N chosen classes.
    query_class = random.choice(chosen_classes)
    query_class_idx = chosen_classes.index(query_class)
    idx_pool = class_to_indices[query_class]
    query_index = random.choice(idx_pool)
    query_img = images[query_index].unsqueeze(0)
    query_lbl = torch.tensor([query_class_idx], dtype=torch.long, device=device)
    support_images = torch.stack(sup_imgs, dim=0)
    support_labels = torch.stack(sup_lbls, dim=0)
    return support_images, support_labels, query_img, query_lbl

##############################################
# Batch Helper: Convert multiple episodes to a batch.
##############################################
def batch_for_few_shot(dataset, n_way, k_shot, batch_size, device="cuda"):
    """
    Samples 'batch_size' episodes and stacks them into a batch.
    
    Returns:
      batch_support: Tensor of shape (B, N*K, 1, 28, 28)
      batch_sup_labels: Tensor of shape (B, N*K, N)
      batch_query: Tensor of shape (B, 1, 1, 28, 28)
      batch_query_labels: Tensor of shape (B,)
    """
    support_list = []
    sup_label_list = []
    query_list = []
    query_label_list = []
    for _ in range(batch_size):
        sup_imgs, sup_lbls, qry_img, qry_lbl = sample_episode(dataset, n_way, k_shot, device)
        support_list.append(sup_imgs)         # (N*K, 1, 28, 28)
        sup_label_list.append(sup_lbls)         # (N*K, N)
        query_list.append(qry_img)              # (1, 1, 28, 28)
        query_label_list.append(qry_lbl)        # (1,)
    batch_support = torch.stack(support_list, dim=0)
    batch_sup_labels = torch.stack(sup_label_list, dim=0)
    batch_query = torch.stack(query_list, dim=0)
    batch_query_labels = torch.cat(query_label_list, dim=0)
    return batch_support, batch_sup_labels, batch_query, batch_query_labels

##############################################
# Training Routine
##############################################
def train_snail(epochs=100, iterations=10000, n_way=5, k_shot=1, batch_size=32, lr=1e-4, device="cuda"):
    """
    Train SNAILFewShot on Omniglot in a batched episodic manner.
    - Uses the precomputed omniglot90_train.pt and omniglot90_test.pt files.
    - For each iteration, a batch of episodes (with 'batch_size' episodes) is sampled.
    - Each episode is an N-way, K-shot task with one query (as the final timestep).
    - The model’s forward method processes the batch and outputs a prediction for each episode (taken from the last timestep).
    - Loss (cross-entropy) is computed only on these final outputs.
    - Training runs for 'iterations' iterations over a fixed number of epochs.
    """
    # Load train and test datasets.
    train_ds, test_ds = load_omniglot_train_test(root="omniglot", download=True)
    
    # Determine maximum sequence length: T = N*K + 1.
    seq_len = n_way * k_shot + 1

    # Initialize model. (Our model’s forward method is assumed to handle a batch by iterating over episodes.)
    model = SNAILFewShot(num_classes=n_way, seq_len=seq_len, device=device, embedding_type="omniglot").to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop.
    # We iterate for a fixed number of iterations (episodes) over a number of epochs.
    total_iterations = iterations
    iter_count = 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(range(0, total_iterations // epochs), desc=f"Epoch {epoch+1}/{epochs}", unit="episode")
        for _ in pbar:
            # Sample a batch of episodes.
            batch_support, batch_sup_labels, batch_query, batch_query_labels = batch_for_few_shot(
                train_ds, n_way, k_shot, batch_size, device)
            # Process each episode in the batch.
            # (Our SNAILFewShot forward method currently handles one episode; we loop over the batch.)
            outputs = []
            for i in range(batch_size):
                sup = batch_support[i]        # shape: (N*K, 1, 28, 28)
                sup_lbl = batch_sup_labels[i]   # shape: (N*K, N)
                qry = batch_query[i]            # shape: (1, 1, 28, 28)
                out = model(sup, sup_lbl, qry)    # shape: (1, n_way)
                outputs.append(out)
            outputs = torch.cat(outputs, dim=0)  # (batch_size, n_way)
            loss = loss_fn(outputs, batch_query_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=-1)
            acc = (preds == batch_query_labels).float().mean().item() * 100
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2f}%", "k_shot": k_shot})
            iter_count += 1

        print(f"Epoch {epoch+1} complete. Total iterations so far: {iter_count}")
    
    # Evaluation on test episodes (using 1-shot for evaluation)
    model.eval()
    test_episodes = 200
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(test_episodes):
            sup, sup_lbl, qry, qry_lbl = sample_episode(test_ds, n_way, 1, device)
            out = model(sup, sup_lbl, qry)
            pred = torch.argmax(out, dim=-1)
            correct += (pred == qry_lbl).sum().item()
            total += 1
    print(f"\n[Test] {n_way}-way, 1-shot => Accuracy: {100.0 * correct / total:.2f}%")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train SNAIL on Omniglot with batched episodes")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--iterations", type=int, default=10000, help="Total episodes/iterations")
    parser.add_argument("--n_way", type=int, default=5, help="N-way classification")
    parser.add_argument("--k_shot", type=int, default=1, help="K-shot support examples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (episodes per update)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cpu or cuda")
    args, unknown = parser.parse_known_args()  # This ignores any unknown arguments.
    return args


def main():
    args = parse_args()
    model = train_snail(epochs=args.epochs, iterations=args.iterations, 
                        n_way=args.n_way, k_shot=args.k_shot, 
                        batch_size=args.batch_size, lr=args.lr, device=args.device)
    # Optionally, save the model state_dict here.
    torch.save(model.state_dict(), os.path.join("omniglot_model.pth"))
    print("Training complete. Model saved as omniglot_model.pth")

if __name__ == "__main__":
    main()

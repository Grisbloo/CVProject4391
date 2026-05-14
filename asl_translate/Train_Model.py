# TL;DR This program takes data from ASL Dataset and uses the .npy landmarks to allow the LSTM to classify the different gestures into their respective alphabetical classes. The metrics that are used is validation
# loss, training loss, and accuracy. During training a slight amount of noise is added to provide the frozen frames some slight variation from the frame they were frozen to, thus allowing use.
# The train_model.py is set to run 50 epochs, only stopping early when 10 epochs do not provide any further better metric. It then saves to best_asl_path to be used during Live.py's deployment.


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from ASL_Model import ASLSequenceInterpreter

# Config
DATA_PATH = os.path.join('ASL_Dataset')
# Matching Actions
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
sequence_length = 16

# If val loss doesn't improve for early_stop_patience epochs, stop early
EARLY_STOP_PATIENCE = 10


# DATASET
class ASLDataset(Dataset):
    def __init__(self, data_path, actions, augment=False):
        self.sequences = []
        self.labels = []
        self.augment = augment  # Only add noise during training, not validation

        for label_num, action in enumerate(actions):
            action_path = os.path.join(data_path, action)
            if not os.path.exists(action_path):
                print(f"  [!] Warning: No folder found for letter '{action}', skipping.")
                continue

            for file_name in os.listdir(action_path):
                if file_name.endswith('.npy'):
                    res = np.load(os.path.join(action_path, file_name))
                    self.sequences.append(res)
                    self.labels.append(label_num)

        print(f"  Loaded {len(self.sequences)} total sequences across {len(set(self.labels))} letters.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy().astype(np.float32)

        # DATA AUGMENTATION
        # Add tiny random noise to keypoints during training only.
        # This prevents the model from memorizing exact coordinates and help generalize to slightly different hand positions and sizes.
        # 0.005 is small enough not to distort the sign, big enough to matter.
        if self.augment:
            noise = np.random.normal(0, 0.005, seq.shape).astype(np.float32)
            seq = seq + noise

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# Load and Split data into specified folders
print("Loading dataset...")
full_dataset = ASLDataset(DATA_PATH, actions, augment=False)

if len(full_dataset) == 0:
    print("No data found! Make sure ASL_Dataset/ has letter folders with .npy files.")
    exit()

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

# Re-wrap the training subset with augmentation ON
train_indices = train_subset.indices
test_indices = test_subset.indices

class IndexedSubset(Dataset):
    def __init__(self, dataset, indices, augment=False):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq, label = self.dataset[self.indices[idx]]
        if self.augment:
            noise = torch.randn_like(seq) * 0.005
            seq = seq + noise
        return seq, label

train_dataset = IndexedSubset(full_dataset, train_indices, augment=True)
test_dataset  = IndexedSubset(full_dataset, test_indices,  augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

print(f"  Train: {len(train_dataset)} samples | Val: {len(test_dataset)} samples")


# Model Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

model     = ASLSequenceInterpreter(num_classes=len(actions)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ReduceLROnPlateau: if val loss stops improving for 5 epochs, cut LR in half.
# This lets the model take big steps early and fine-tune carefully later.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)


# Training Loop
epochs = 50
best_val_loss = float('inf')
epochs_without_improvement = 0  # Early stopping counter

print("\nStarting training...")
print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Val Acc':<12} {'LR'}")
print("-" * 60)

for epoch in range(epochs):

    # Train
    model.train()
    total_train_loss = 0

    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validate
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for val_seq, val_labels in test_loader:
            val_seq, val_labels = val_seq.to(device), val_labels.to(device)
            val_preds = model(val_seq)
            total_val_loss += criterion(val_preds, val_labels).item()

            # Accuracy: count how many predictions matched the true label
            predicted_classes = torch.argmax(val_preds, dim=1)
            correct += (predicted_classes == val_labels).sum().item()
            total += val_labels.size(0)

    avg_val_loss = total_val_loss / len(test_loader)
    val_accuracy = 100.0 * correct / total
    current_lr = optimizer.param_groups[0]['lr']

    # Step the scheduler based on val loss
    scheduler.step(avg_val_loss)

    # Print every epoch to watch for dev debugging
    print(f"{epoch+1:<8} {avg_train_loss:<14.4f} {avg_val_loss:<14.4f} {val_accuracy:<11.1f}% {current_lr:.6f}")

    # Save Best Model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_asl_model.pth')
        print(f"  ✓ New best saved! (val loss: {avg_val_loss:.4f}, acc: {val_accuracy:.1f}%)")
    else:
        epochs_without_improvement += 1

    # Early Stopping
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping triggered — no improvement for {EARLY_STOP_PATIENCE} epochs.")
        break

print("\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("Best model saved as 'best_asl_model.pth'")

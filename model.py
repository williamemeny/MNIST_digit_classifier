# ============================
# MNIST MLP — fully commented
# ============================

import torch                              # PyTorch core tensor & autograd library
import torch.nn as nn                     # Neural network modules (Linear, Conv, etc.)
import torch.nn.functional as F           # Functional ops (relu, softmax, etc.)
from torch.utils.data import DataLoader   # Efficient batching/iteration over datasets
from torchvision import datasets, transforms  # Standard CV datasets & preprocessing

# ----- 1) Config -----
BATCH_SIZE = 128                          # how many images per training step
EPOCHS = 10                                # full passes over the training set
LR = 1e-3                                 # learning rate (step size for optimizer)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if available
torch.manual_seed(0)                      # make weight init & shuffles reproducible

# ----- 2) Data: MNIST (28x28 grayscale) -----
# ToTensor(): converts PIL image HxWxC in [0,255] -> tensor CxHxW in [0,1]
# Normalize(): per-channel (here single channel) standardization: (x - mean) / std
transform = transforms.Compose([
    transforms.ToTensor(),                # -> shape [1, 28, 28], values in [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
])

train_ds = datasets.MNIST(                # download & prepare training set
    root="./data", train=True, download=True, transform=transform
)
test_ds  = datasets.MNIST(                # download & prepare test set
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(                # mini-batch iterator over train_ds
    train_ds, batch_size=BATCH_SIZE, shuffle=True,  # shuffle for SGD
    num_workers=0, pin_memory=False                  # speed-ups on GPU systems
)
test_loader  = DataLoader(                # mini-batch iterator over test_ds
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=False
)

# ----- 3) Model: MLP -----
# We flatten each 28x28 image -> vector of length 784, then pass through:
# Linear(784->512) -> ReLU -> Dropout -> Linear(512->256) -> ReLU -> Dropout -> Linear(256->10)
# Output are logits for 10 classes (digits 0..9)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()                # sets up nn.Module internals
        self.fc1 = nn.Linear(28*28, 512)  # first fully-connected layer
        self.fc2 = nn.Linear(512, 256)    # second fully-connected layer
        self.fc3 = nn.Linear(256, 10)     # final layer -> 10 class logits
        self.dropout1 = nn.Dropout(0.3)   # Higher dropout for first layer (more parameters)
        self.dropout2 = nn.Dropout(0.2)   # Lower dropout for second layer
        # No dropout after final layer (output layer)

    def forward(self, x):
        # x arrives as a batch of images: shape [B, 1, 28, 28]
        x = x.view(x.size(0), -1)         # flatten to [B, 784]
        x = F.relu(self.fc1(x))           # [B, 512], nonlinearity adds capacity
        x = self.dropout1(x)               # 30% dropout after first layer
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)               # 20% dropout after second layer
        logits = self.fc3(x)              # No dropout before output
        return logits                     # leave as logits; CE loss applies softmax internally

model = MLP().to(DEVICE)                  # move parameters to GPU/CPU

# ----- 4) Optimizer and Loss -----
optimizer = torch.optim.AdamW(            # AdamW: adaptive LR + decoupled weight decay
    model.parameters(), lr=LR
)
criterion = nn.CrossEntropyLoss()         # CE = LogSoftmax + NLLLoss under the hood
# expects: logits of shape [B, C] and integer class labels [B]

# ----- 5) Training loop -----
def train_one_epoch(epoch: int):
    model.train()                         # enable training mode (e.g., dropout if present)
    total_loss, total_correct, total_examples = 0.0, 0, 0
    
    # Calculate progress intervals (every 10% of batches)
    total_batches = len(train_loader)
    progress_interval = max(1, total_batches // 10)  # Update every 10% of batches
    
    print(f"\nEpoch {epoch} - Training Progress:")
    print("=" * 50)
    
    for batch_idx, (images, labels) in enumerate(train_loader):   # iterate mini-batches
        images = images.to(DEVICE, non_blocking=True)  # move batch to device
        labels = labels.to(DEVICE, non_blocking=True)  # shape [B], ints 0..9

        optimizer.zero_grad()             # clear stale gradients (PyTorch accumulates by default)
        logits = model(images)            # forward pass -> [B, 10]
        loss = criterion(logits, labels)  # CE compares logits vs true labels -> scalar
        loss.backward()                   # autograd computes d(loss)/d(params)
        optimizer.step()                  # apply parameter updates (one SGD/AdamW step)

        total_loss += loss.item() * images.size(0)      # sum batch loss * batch size
        preds = logits.argmax(dim=1)                     # predicted class index per sample
        total_correct += (preds == labels).sum().item()  # count correct predictions
        total_examples += images.size(0)                 # track dataset coverage
        
        # Progress update every 10% of batches
        if (batch_idx + 1) % progress_interval == 0 or batch_idx == total_batches - 1:
            progress_percent = ((batch_idx + 1) / total_batches) * 100
            current_loss = total_loss / total_examples
            current_acc = total_correct / total_examples
            current_batch_loss = loss.item()
            
            print(f"  {progress_percent:5.1f}% | "
                  f"Batch Loss: {current_batch_loss:.4f} | "
                  f"Running Loss: {current_loss:.4f} | "
                  f"Running Acc: {current_acc:.4f} | "
                  f"Batch {batch_idx + 1}/{total_batches}")

    avg_loss = total_loss / total_examples               # mean loss over all samples
    acc = total_correct / total_examples                 # training accuracy
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Final Train Loss: {avg_loss:.4f}")
    print(f"  Final Train Acc:  {acc:.4f}")
    return avg_loss, acc

# ----- 6) Evaluation -----
@torch.no_grad()                        # disable gradient tracking for speed/memory
def evaluate():
    model.eval()                        # eval mode (e.g., disables dropout, fixes BN stats)
    total_correct, total_examples = 0, 0
    
    print("\nEvaluating on Test Set:")
    print("-" * 30)
    
    # Calculate progress intervals for evaluation too
    total_test_batches = len(test_loader)
    progress_interval = max(1, total_test_batches // 10)
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(DEVICE, non_blocking=True)   # move batch to device
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)                          # forward only
        preds = logits.argmax(dim=1)                    # pick highest logit per sample
        total_correct += (preds == labels).sum().item() # accumulate corrects
        total_examples += images.size(0)                # accumulate counts
        
        # Progress update every 10% of test batches
        if (batch_idx + 1) % progress_interval == 0 or batch_idx == total_test_batches - 1:
            progress_percent = ((batch_idx + 1) / total_test_batches) * 100
            current_acc = total_correct / total_examples
            print(f"  {progress_percent:5.1f}% | "
                  f"Running Test Acc: {current_acc:.4f} | "
                  f"Batch {batch_idx + 1}/{total_test_batches}")

    acc = total_correct / total_examples                # test accuracy (generalization)
    print(f"\nFinal Test Accuracy: {acc:.4f}")
    return acc

# ----- 7) Run -----
if __name__== '__main__':
    final_train_loss = 0.0
    final_test_accuracy = 0.0
    
    # Lists to store epoch statistics for summary table
    epoch_stats = []
    
    for epoch in range(1, EPOCHS + 1):      # iterate epochs: 1..EPOCHS inclusive
        train_loss, train_acc = train_one_epoch(epoch)  # train for one pass over training data
        test_acc = evaluate()                           # evaluate on held-out test set
        
        # Store statistics for this epoch
        epoch_stats.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': train_acc - test_acc  # Training vs Test accuracy difference
        })
        
        # Store the final epoch values
        if epoch == EPOCHS:
            final_train_loss = train_loss
            final_test_accuracy = test_acc
    
    # Print comprehensive summary table
    print("\n" + "="*100)
    print("TRAINING SUMMARY TABLE")
    print("="*100)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Trend':<6} {'Train Acc':<12} {'Trend':<6} {'Test Acc':<12} {'Trend':<6} {'Gap':<8} {'Trend':<6}")
    print("-"*100)
    
    for i, stats in enumerate(epoch_stats):
        # Determine trend arrows for each metric
        if i == 0:
            loss_trend = "─"    # First epoch, no trend
            train_trend = "─"   # First epoch, no trend
            test_trend = "─"    # First epoch, no trend
            gap_trend = "─"     # First epoch, no trend
        else:
            prev_stats = epoch_stats[i-1]
            
            # Train Loss trend (lower is better)
            if stats['train_loss'] < prev_stats['train_loss']:
                loss_trend = "↘️"  # Loss decreasing (good)
            elif stats['train_loss'] > prev_stats['train_loss']:
                loss_trend = "↗️"  # Loss increasing (bad)
            else:
                loss_trend = "→"   # Loss unchanged
            
            # Train Accuracy trend (higher is better)
            if stats['train_acc'] > prev_stats['train_acc']:
                train_trend = "↗️"  # Accuracy increasing (good)
            elif stats['train_acc'] < prev_stats['train_acc']:
                train_trend = "↘️"  # Accuracy decreasing (bad)
            else:
                train_trend = "→"   # Accuracy unchanged
            
            # Test Accuracy trend (higher is better)
            if stats['test_acc'] > prev_stats['test_acc']:
                test_trend = "↗️"  # Accuracy increasing (good)
            elif stats['test_acc'] < prev_stats['test_acc']:
                test_trend = "↘️"  # Accuracy decreasing (bad)
            else:
                test_trend = "→"   # Accuracy unchanged
            
            # Gap trend (lower is better for generalization)
            if stats['gap'] < prev_stats['gap']:
                gap_trend = "↘️"  # Gap decreasing (good)
            elif stats['gap'] > prev_stats['gap']:
                gap_trend = "↗️"  # Gap increasing (bad)
            else:
                gap_trend = "→"   # Gap unchanged
        
        print(f"{stats['epoch']:<6} "
              f"{stats['train_loss']:<12.4f} "
              f"{loss_trend:<6} "
              f"{stats['train_acc']:<12.4f} "
              f"{train_trend:<6} "
              f"{stats['test_acc']:<12.4f} "
              f"{test_trend:<6} "
              f"{stats['gap']:<8.4f} "
              f"{gap_trend:<6}")
    
    print("-"*100)
    
    # Overall statistics
    best_test_acc = max(stats['test_acc'] for stats in epoch_stats)
    best_epoch = next(stats['epoch'] for stats in epoch_stats if stats['test_acc'] == best_test_acc)
    avg_gap = sum(stats['gap'] for stats in epoch_stats) / len(epoch_stats)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Best Test Accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")
    print(f"  Average Train-Test Gap: {avg_gap:.4f}")
    print(f"  Final Train-Test Gap: {epoch_stats[-1]['gap']:.4f}")
    
    # Save the trained model with tracked values
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'train_loss': final_train_loss,        
        'test_accuracy': final_test_accuracy,
        'epoch_stats': epoch_stats,  # Save all epoch statistics
        'best_test_accuracy': best_test_acc,
        'best_epoch': best_epoch
    }, 'mnist_classifier.pth')
    print("Model saved as 'mnist_classifier.pth'")
    


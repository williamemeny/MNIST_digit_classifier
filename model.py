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
EPOCHS = 5                                # full passes over the training set
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

    for images, labels in train_loader:   # iterate mini-batches
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

    avg_loss = total_loss / total_examples               # mean loss over all samples
    acc = total_correct / total_examples                 # training accuracy
    print(f"Epoch {epoch}: train loss {avg_loss:.4f}, acc {acc:.4f}")
    return avg_loss, acc  # Return the actual values

# ----- 6) Evaluation -----
@torch.no_grad()                        # disable gradient tracking for speed/memory
def evaluate():
    model.eval()                        # eval mode (e.g., disables dropout, fixes BN stats)
    total_correct, total_examples = 0, 0

    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)   # move batch to device
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)                          # forward only
        preds = logits.argmax(dim=1)                    # pick highest logit per sample
        total_correct += (preds == labels).sum().item() # accumulate corrects
        total_examples += images.size(0)                # accumulate counts

    acc = total_correct / total_examples                # test accuracy (generalization)
    print(f"Test accuracy: {acc:.4f}")
    return acc  # Return the actual accuracy

# ----- 7) Run -----
if __name__== '__main__':
    final_train_loss = 0.0
    final_test_accuracy = 0.0
    
    for epoch in range(1, EPOCHS + 1):      # iterate epochs: 1..EPOCHS inclusive
        train_loss, train_acc = train_one_epoch(epoch)  # train for one pass over training data
        test_acc = evaluate()                           # evaluate on held-out test set
        
        # Store the final epoch values
        if epoch == EPOCHS:
            final_train_loss = train_loss
            final_test_accuracy = test_acc
    
    # Save the trained model with tracked values
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'train_loss': final_train_loss,        
        'test_accuracy': final_test_accuracy,  
    }, 'mnist_classifier.pth')
    print("Model saved as 'mnist_classifier.pth'")
    


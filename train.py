import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os
from tqdm import tqdm
import glob
import sys
from torch.utils.data import random_split

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second block
        self.conv4 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third block - Moved pooling closer to prediction
        self.conv7 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=1)  # Removed stride=2
        self.pool3 = nn.MaxPool2d(2, 2)  # Added pooling after conv9
        self.bn9 = nn.BatchNorm2d(32)  # Moved BN after pooling

        # Modified dropout strategy
        self.dropout1 = nn.Dropout(0.05)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.25)
        
        self.fc = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout2(x)
        
        # Third block - Modified order
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.conv9(x))
        x = self.pool3(x)  # Pooling closer to prediction
        x = self.bn9(x)    # BN after pooling
        x = self.dropout3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy

def clean_old_models():
    """Delete all previously saved model files"""
    model_files = glob.glob('model_*.pth')
    for f in model_files:
        try:
            os.remove(f)
            print(f"Deleted old model: {f}")
        except OSError as e:
            print(f"Error deleting {f}: {e}")

def is_ci_environment():
    """Check if we're running in a CI environment"""
    return 'CI' in os.environ

def calculate_normalization_values(dataset):
    """Calculate mean and std of the dataset"""
    loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(dataset)
    std /= len(dataset)
    return mean.item(), std.item()

def train():
    clean_old_models()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # First load dataset without normalization to calculate values
    base_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    temp_dataset = datasets.MNIST('data', train=True, download=True, transform=base_transform)
    mean, std = calculate_normalization_values(temp_dataset)
    print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Now create transforms with calculated normalization values
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Additional augmentation
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    # Use official MNIST train/test split with new transforms
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
    val_dataset = datasets.MNIST('data', train=False, download=True, transform=transform_test)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000)
    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    num_epochs = 19
    best_accuracy = 0
    best_model_path = None
    
    lr_schedule = {
        8: 0.05,   # First reduction at epoch 8
        12: 0.01,  # Second reduction at epoch 12
        15: 0.005  # Final reduction at epoch 15
    }
    
    for epoch in range(num_epochs):
        # Adjust learning rate according to schedule
        if epoch in lr_schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[epoch]
                print(f'Learning rate adjusted to: {lr_schedule[epoch]}')
        
        # Training
        model.train()
        total_batches = len(train_loader)
        
        if is_ci_environment():
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}: Progress {batch_idx}/{total_batches} batches', flush=True)
        else:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{current_lr:.6f}'
                })
        
        # Evaluate after each epoch
        accuracy = evaluate(model, device, val_loader)
        print(f'Epoch {epoch+1}/{num_epochs} - Test Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if best_model_path:
                try:
                    os.remove(best_model_path)
                except:
                    pass
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            accuracy_str = f"{accuracy:.2f}".replace(".", "p")
            best_model_path = f'model_{timestamp}_acc{accuracy_str}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with accuracy: {accuracy:.2f}%')
    
    print(f'\nBest Test Accuracy: {best_accuracy:.2f}%')
    return best_model_path

if __name__ == "__main__":
    save_path = train()
    print(f"\nModel saved to: {save_path}")
    
    # Enhanced model summary
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Architecture Summary:")
    print("=" * 40)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    if not is_ci_environment():
        from torchsummary import summary
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        summary(model, input_size=(1, 28, 28))
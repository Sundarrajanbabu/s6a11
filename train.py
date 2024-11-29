import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.pool1(F.relu(self.bn3(self.conv3(F.relu(self.bn1(self.conv1(x)))))))
        x = self.dropout(x)
        x = self.pool2(F.relu(self.bn6(self.conv6(F.relu(self.bn4(self.conv4(x)))))))
        x = self.dropout(x)
        x = F.relu(self.bn9(self.conv9(F.relu(self.bn7(self.conv7(x))))))
        x = self.dropout(x)
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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load both training and test datasets
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())
    
    # Training
    model.train()
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    # Evaluate
    accuracy = evaluate(model, device, test_loader)
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_str = f"{accuracy:.2f}".replace(".", "p")
    save_path = f'model_{timestamp}_acc{accuracy_str}.pth'
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    save_path = train()
    print(f"\nModel saved to: {save_path}")
    
    # Model summary
    from torchsummary import summary
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28)) 
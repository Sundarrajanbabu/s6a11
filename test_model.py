import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from train import Net
import pytest
import glob
import os

def get_latest_model():
    model_files = glob.glob('model_*.pth')
    return max(model_files, key=os.path.getctime)

def test_model_architecture():
    model = Net()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"
    
    # Check for batch normalization layers
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should have batch normalization layers"
    
    # Check for dropout
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should have dropout layers"
    
    # Check for fully connected layer
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert has_fc, "Model should have fully connected layers"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(get_latest_model()))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
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
    assert accuracy > 99, f"Model accuracy {accuracy}% is less than 99%"

if __name__ == "__main__":
    pytest.main([__file__]) 
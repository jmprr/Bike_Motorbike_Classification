import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from dataset import BikeBicycleDataset
from torchvision import transforms
import shutil
from tqdm import tqdm
from model import resnet18

transform_train = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = BikeBicycleDataset(root_dir='./Bike_and_motorbike', transform=transform_train, train=True)
val_dataset = BikeBicycleDataset(root_dir='./Bike_and_motorbike', transform=transform_val, train=False)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Model
model = resnet18(num_classes=2) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_model(model, filename='final_model.pth'):
    torch.save(model.state_dict(), filename)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch+1}/{num_epochs}')

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update the progress bar
            pbar.update(inputs.shape[0])
            pbar.set_postfix({'loss': running_loss / total, 'accuracy': 100. * correct / total})
        
        pbar.close()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        val_loss, val_acc = validate(model, val_loader, criterion)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
        }, is_best)

        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

    save_model(model, 'final_model.pth')

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)


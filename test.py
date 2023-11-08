import torch
from PIL import Image
from torchvision import transforms
from model import resnet18
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = resnet18(num_classes=2)  
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_model(checkpoint_path, num_classes):
    model = resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# model = load_checkpoint('./model_best.pth.tar')
# model.eval()  # Set the model to evaluation mode

model = load_model('./final_model.pth', num_classes=2)

def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image


idx_to_class = {0: 'bike', 1: 'motorbike'}

def predict_image(model, image_path, class_mapping):
    image = prepare_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_idx = preds[0].item()
    return class_mapping[predicted_idx]


folder_path = './Test_image'
for filename in os.listdir(folder_path):
    print(filename)
    image_path = os.path.join(folder_path, filename)
    prediction = predict_image(model, image_path, idx_to_class)
    print(f'Predicted class: {prediction}')

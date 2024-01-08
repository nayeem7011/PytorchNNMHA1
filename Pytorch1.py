import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Neural Network Model Definition
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(65536, 64)  # Adjust this based on your image size and network architecture
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Adjust size based on your model input size
    transforms.ToTensor(),
])

# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f'rotated_rectangle_{idx}.png')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        angle = self.data['angle'][idx]
        angle = torch.tensor([angle], dtype=torch.float32)
        return image, angle

# Instantiate Model, Loss Function, and Optimizer
model = MyCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dataset and DataLoader
image_directory = 'C:/Users/Asus/OneDrive/Desktop/NNMHA/generated_images'
csv_file = 'C:/Users/Asus/OneDrive/Desktop/NNMHA/rectangle_angles.csv'
my_dataset = MyDataset(csv_file=csv_file, img_dir=image_directory, transform=transform)
dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Save the Model
torch.save(model.state_dict(), 'C:/Users/Asus/OneDrive/Desktop/NNMHA_model.pth')

# Load the Model for Inference
model.load_state_dict(torch.load('C:/Users/Asus/OneDrive/Desktop/NNMHA_model.pth'))
model.eval()

# Inference on a Sample Image
sample_data = pd.read_csv(csv_file).sample(n=1)
idx = sample_data.index[0]  # Get the index of the sampled data
image_path = os.path.join(image_directory, f'rotated_rectangle_{idx}.png')
label = sample_data.iloc[0]['angle']

input_image = Image.open(image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

# Perform Inference
with torch.no_grad():
    output = model(input_tensor)

# Print Results
print("Actual Label:", label)
print("Model Prediction:", output.item())

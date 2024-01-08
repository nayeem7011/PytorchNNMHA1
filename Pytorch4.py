import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Data Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, start_idx=0, end_idx=None):
        self.data = pd.read_csv(csv_file)
        if end_idx is not None:
            self.data = self.data.iloc[start_idx:end_idx].reset_index(drop=True)
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

# Splitting dataset
total_images = 1000
train_images = int(total_images * 0.8)

train_dataset = MyDataset(csv_file=csv_file, img_dir=image_directory, transform=transform, end_idx=train_images)
test_dataset = MyDataset(csv_file=csv_file, img_dir=image_directory, transform=transform, start_idx=train_images, end_idx=total_images)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch in train_dataloader:
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



# Evaluate on Test Data
model.eval()  # Set the model to evaluation mode

# Lists to store actual and predicted angles
actual_angles = []
predicted_angles = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        predicted_angle = outputs.item()  # Get the predicted angle
        actual_angle = labels.item()  # Get the actual angle

        # Append to lists
        predicted_angles.append(predicted_angle)
        actual_angles.append(actual_angle)

        print(f'Actual angle: {actual_angle}, Predicted angle: {predicted_angle}')

# Optionally, you can calculate and print the average error
average_error = sum(abs(a - p) for a, p in zip(actual_angles, predicted_angles)) / len(actual_angles)
print(f'Average error: {average_error:.4f}')



mse = mean_squared_error(actual_angles, predicted_angles)
print(f'Mean Squared Error: {mse:.4f}')


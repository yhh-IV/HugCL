import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from predictor import WaleNet

class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super(GaussianNLLLoss, self).__init__()

    def forward(self, output, target):
        mu_x = output[..., 0]
        mu_y = output[..., 1]
        
        sigma_x = torch.exp(torch.clamp(output[..., 2], min=-1, max=1))
        sigma_y = torch.exp(torch.clamp(outputs[..., 3], min=-1, max=1))
        sigma_xy = torch.clamp(outputs[..., 4], min=-0.5, max=0.5)
        
        diff_x = target[..., 0] - mu_x
        diff_y = target[..., 1] - mu_y
        
        det = 1 / (2 * (1 - sigma_xy ** 2))
        
        loss = torch.log(sigma_x) + torch.log(sigma_y) + 0.5*torch.log(1-sigma_xy**2) + \
               det * ((diff_x/sigma_x)**2 + (diff_y/sigma_y)**2 + 2*sigma_xy*diff_x*diff_y/(sigma_x*sigma_y)) 
               
        return torch.mean(loss)

class TrajectoryDataset(Dataset):
    def __init__(self, input_data_list, output_data_list):
        self.input_list = input_data_list
        self.output_list = output_data_list
    
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):
        inputs = self.input_list[idx]
        outputs = self.output_list[idx]
        return inputs, outputs


# Hyperparameters
input_dim = 4  # Input feature dimension (x, y, yaw, v)
hidden_dim = 128  # LSTM hidden state dimension
num_layers = 4  # Number of LSTM layers
output_dim = 5  # Output dimension (x, y, σx, σy, σxy)
output_timesteps = 20  # Number of output timesteps

input_data_list, output_data_list = [], []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

traj_path = '../traj_data.pkl'
with open(traj_path, 'rb') as file:
    traj_data = pickle.load(file)[1:]

for i in range(len(traj_data)):
    for j in range(1, len(traj_data[i])):
        input_data = np.array((traj_data[i][j][4][0], traj_data[i][j][5][0], traj_data[i][j][6][0], traj_data[i][j][7][0]))
        output_data = np.array((traj_data[i][j][4], traj_data[i][j][5])) - np.array([[traj_data[i][j][4][0], traj_data[i][j][5][0]]]).T
        input_data_list.append(input_data)
        output_data_list.append(output_data.T)

# Create dataset
dataset = TrajectoryDataset(input_data_list, output_data_list)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create model
model = WaleNet(input_dim, hidden_dim, num_layers, output_dim, output_timesteps).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = GaussianNLLLoss()

# Trian model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_mae_s = []
    total_mae_d = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs.repeat(1, output_timesteps, 1).float())
        
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        mae_s = torch.mean(torch.abs(outputs[:, :, 0] - targets[:, :, 0]))
        mae_d = torch.mean(torch.abs(outputs[:, :, 1] - targets[:, :, 1]))
        total_mae_s.append(mae_s.item())
        total_mae_d.append(mae_d.item())
        
    epoch_loss = running_loss / len(dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    print("Ave. MAE_s, MAE_d:", np.mean(total_mae_s), np.mean(total_mae_d))
    
print("Training complete.")

model_path = './predictor.pth'
torch.save(model.state_dict(), model_path)

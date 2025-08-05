import torch
import torch.nn as nn

# Predict the trajectories of surrounding road users
class WaleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, output_timesteps):
        super(WaleNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_timesteps = output_timesteps

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out
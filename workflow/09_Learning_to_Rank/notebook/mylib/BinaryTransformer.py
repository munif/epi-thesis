import numpy as np
import torch
import torch.nn as nn

class BinaryTransformer1D(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(BinaryTransformer1D, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(0)  # Remove sequence dimension
        x = self.fc(x)
        return x

def transform_feature(x, model, device):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    
    model.eval()
    x_1d = model(x_tensor)

    return x_1d.detach().cpu().numpy()
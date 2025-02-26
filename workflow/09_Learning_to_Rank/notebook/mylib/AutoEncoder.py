import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            
            nn.Linear(input_dim, bottleneck_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            
            nn.Linear(input_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AutoEncoderSmall(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, hidden_layer=[256, 128, 64]):
        super(AutoEncoderSmall, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layer[0]),
            nn.BatchNorm1d(hidden_layer[0]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_layer[0], hidden_layer[1]),
            nn.BatchNorm1d(hidden_layer[1]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_layer[1], bottleneck_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_layer[1]),
            nn.BatchNorm1d(hidden_layer[1]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_layer[1], hidden_layer[0]),
            nn.BatchNorm1d(hidden_layer[0]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_layer[0], input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class LinearAutoEncoder(nn.Module):
    def __init__(self, input_dim=20000, bottleneck_dim=100):
        super(LinearAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First layer
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            # Second layer
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            # Third layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Final layer
            nn.Linear(256, bottleneck_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # First layer
            nn.Linear(bottleneck_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Second layer
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            # Third layer
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            # Final layer
            nn.Linear(1024, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
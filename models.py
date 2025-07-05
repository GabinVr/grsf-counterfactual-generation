import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import gen 

import numpy as np
import math
import wildboar.datasets as wb_datasets
import wildboar.distance as wb_distance
import torch
import torch.nn as nn
from pyts.approximation import PiecewiseAggregateApproximation as PAA

class SimpleFeedforwardClassifier(gen.BaseSurrogateClassifier):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(SimpleFeedforwardClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        if len(x.shape) == 3:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1) 

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    @staticmethod
    def get_params(sample_size: int = None, num_classes: int = None):
        """Return model parameters."""
        params = {
            "input_size": {"type": "int", "default": 1, "description": "Should be the length of the input sequence"},
            "hidden_size": {"type": "int", "default": 100, "description": "Size of the hidden layer"},
            "output_size": {"type": "int", "default": 2, "description": "Number of output classes"},
        }
        if sample_size is not None:
            params["input_size"]["default"] = sample_size
        if num_classes is not None:
            params["output_size"]["default"] = num_classes
        return params
    
class LSTMClassifier(gen.BaseSurrogateClassifier):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout:float=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Reshape input to (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Initialize hidden state and cell state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        h_n, c_n = self.lstm(x, (h_0, c_0))
        h_n = h_n.view(-1, self.hidden_size)
        x = self.fc(h_n)
        return x
    
    @staticmethod
    def get_params(sample_size: int = None, num_classes: int = None):
        """Return model parameters."""
        params = {
            "input_size": {"type": "int", "default": 1, "description": "Should be the length of the input sequence"},
            "hidden_size": {"type": "int", "default": 100, "description": "Size of the hidden layer"},
            "num_layers": {"type": "int", "default": 1, "description": "Number of LSTM layers"},
            "output_size": {"type": "int", "default": 2, "description": "Number of output classes"},
            "dropout": {"type": "float", "default": 0.2, "description": "Dropout rate for LSTM layers"},
        }

        if sample_size is not None:
            params["input_size"]["default"] = sample_size
        if num_classes is not None:
            params["output_size"]["default"] = num_classes
        return params

class CNNClassifier(gen.BaseSurrogateClassifier):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn = nn.Conv1d(1, hidden_size, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size * input_size, output_size)
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Reshape input to (batch_size, channels, sequence_length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Apply CNN
        x = self.cnn(x)
        x = self.relu(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    @staticmethod
    def get_params(sample_size: int = None, num_classes: int = None):
        """Return model parameters."""
        params = {
            "input_size": {"type": "int", "default": 1, "description": "Should be the length of the input sequence"},
            "hidden_size": {"type": "int", "default": 50, "description": "Size of the hidden layer"},
            "output_size": {"type": "int", "default": 2, "description": "Number of output classes"},
        }
        if sample_size is not None:
            params["input_size"]["default"] = sample_size
        if num_classes is not None:
            params["output_size"]["default"] = num_classes
        return params

class TransformerClassifier(gen.BaseSurrogateClassifier):
    def __init__(self,
                 timestep: int,
                 num_classes: int, 
                 d_model=256,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=512,
                 dropout=0.3):
        super(TransformerClassifier, self).__init__()

        self.d_model = d_model
        self.timestep = timestep
        self.num_classes = num_classes
        self.input_shape = (timestep, 1)  # Pour les séries temporelles 1D

        # AJOUT: Embedding pour convertir 1D vers d_model
        self.embedding = nn.Linear(1, d_model)  # 1 feature par timestep -> d_model

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)

        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape attendue: (batch_size, sequence_length) pour les séries temporelles
        
        if len(x.shape) == 1:
            # (sequence_length,) -> (1, sequence_length, 1)
            x = x.unsqueeze(0).unsqueeze(-1)
        elif len(x.shape) == 2:
            # (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
            x = x.unsqueeze(-1)
        elif len(x.shape) == 3:
            # Cas où gen.py a fait unsqueeze(1): (batch_size, 1, sequence_length)
            if x.shape[1] == 1:
                # Transpose pour avoir (batch_size, sequence_length, 1)
                x = x.transpose(1, 2)

        batch_size, seq_len, features = x.shape
        
        # Embedding: (batch_size, sequence_length, 1) -> (batch_size, sequence_length, d_model)
        x = self.embedding(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, sequence_length, d_model)
        
        x = x.mean(dim=1)  
        x = self.dropout(x)  
        
        # Classification finale
        x = self.classifier(x)  # (batch_size, num_classes)
        
        return x

    @staticmethod
    def get_params(sample_size: int = None, num_classes: int = None):
        """Return model parameters."""
        params = {
            "timestep": {"type": "int", "default": 128, "description": "Length of the input sequence"},
            "num_classes": {"type": "int", "default": 2, "description": "Number of output classes"},
            "d_model": {"type": "int", "default": 256, "description": "Dimension of the model"},
            "nhead": {"type": "int", "default": 4, "description": "Number of heads in the multihead attention"},
            "num_layers": {"type": "int", "default": 2, "description": "Number of transformer layers"},
            "dim_feedforward": {"type": "int", "default": 512, "description": "Dimension of the feedforward network"},
            "dropout": {"type": "float", "default": 0.3, "description": "Dropout rate for transformer layers"},
        }
        if sample_size is not None:
            params["timestep"]["default"] = sample_size
        if num_classes is not None:
            params["num_classes"]["default"] = num_classes
        return params

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0,1))

    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)

def listModels():
    """
    List available models for counterfactual generation.
    
    Returns:
        list: List of model names.
    """
    return {"LSTMClassifier": LSTMClassifier, 
            "CNNClassifier": CNNClassifier,
            "SimpleFeedforwardClassifier": SimpleFeedforwardClassifier,
            "TransformerClassifier": TransformerClassifier}

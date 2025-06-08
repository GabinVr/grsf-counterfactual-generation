import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import gen 

import numpy as np
import wildboar.datasets as wb_datasets
import wildboar.distance as wb_distance
import torch
import torch.nn as nn
from pyts.approximation import PiecewiseAggregateApproximation as PAA



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
    def get_params():
        """Return model parameters."""
        return {
            "input_size": "Size of the input features",
            "hidden_size": "Size of the hidden layer",
            "num_layers": "Number of LSTM layers",
            "output_size": "Number of output classes",
            "dropout": "Dropout rate (default: 0.2)"
        }

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
    def get_params():
        """Return model parameters."""
        return {
            "input_size": "Should be the length of the input sequence",
            "hidden_size": "Size of the hidden layer",
            "output_size": "Number of output classes",
        }

def listModels():
    """
    List available models for counterfactual generation.
    
    Returns:
        list: List of model names.
    """
    return {"LSTMClassifier": LSTMClassifier, 
            "CNNClassifier": CNNClassifier}

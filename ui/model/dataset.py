
import os
import sys

UI_ROOT = os.path.dirname(os.path.dirname(__file__))
if UI_ROOT not in sys.path:
    sys.path.append(UI_ROOT)
from utils import getDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import gen


class DatasetObject:
    def __init__(self):
        self._dataset_name = None
        self._raw_data = None    
        self._split_data = None  
        self._sample_size = None 
        self._num_classes = None 
        self._is_empty = True

    def is_empty(self) -> bool:
        return self._is_empty
    
    def has_changed(self, dataset_name: str) -> bool:
        return self._dataset_name != dataset_name

    def clear(self) -> None:
        self._dataset_name = None
        self._raw_data = None
        self._split_data = None
        self._sample_size = None
        self._num_classes = None
        self._is_empty = True
    
    def set_dataset(self, dataset_name: str) -> None:
        if not isinstance(dataset_name, str):
            raise ValueError("Dataset name must be a string.")
        self._dataset_name = dataset_name
        self.load_dataset()

    def load_dataset(self) -> None:
        if self._dataset_name is None:
            raise ValueError("Dataset name not set. Please set a dataset name first.")
        
        self._raw_data = getDataset(self._dataset_name)
        if self._raw_data is None:
            raise ValueError(f"Dataset '{self._dataset_name}' not found or could not be loaded.")
        
        X, y = self._raw_data
        self._sample_size = X.shape[1]
        self._num_classes = len(set(y))
        self._is_empty = False

    def set_split_dataset(self, split_dataset: tuple) -> None:
        self._split_data = split_dataset

    def get_dataset(self):
        if self._raw_data is None:
            raise ValueError("Dataset not loaded. Please load a dataset first.")
        return self._raw_data

    def get_dataset_name(self):
        if self._dataset_name is None:
            raise ValueError("Dataset not loaded. Please load a dataset first.")
        return self._dataset_name
    
    def get_split_dataset(self):
        if self._split_data is None:
            raise ValueError("Split dataset not set. Please set a split dataset first.")
        return self._split_data
    
    def get_sample_size(self) -> int:
        if self._sample_size is None:
            raise ValueError("Dataset not loaded. Please load a dataset first.")
        return self._sample_size
    
    def get_num_classes(self) -> int:
        if self._num_classes is not None:
            return self._num_classes
        raise ValueError("Dataset not loaded. Please load a dataset first.")

    def __str__(self) -> str:
        if self._is_empty:
            return "DatasetObject is empty."
            
        info = [
            f"📊 **Dataset**: {self._dataset_name}",
            f"📏 **Nb Samples**: {self._raw_data[0].shape[0]} series",
            f"📈 **Length**: {self._sample_size} timesteps",
            f"🏷️ **Classes**: {self._num_classes} classes",
        ]
        
        return "\n\n".join(info)
    

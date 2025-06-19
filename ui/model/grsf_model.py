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



class GRSFModelObject:
    def __init__(self):
        self._model = None
        self._parameters = None
        self._is_trained = False
        self._split_dataset = None
        self._accuracy = None
    
    def is_empty(self) -> bool:
        return self._model is None
    
    def has_changed(self, parameters: dict) -> bool:
        if self._parameters is None:
            return True

        return self._parameters != parameters

    def clear(self) -> None:
        self._model = None
        self._parameters = None
        self._is_trained = False
        self._split_dataset = None
        self._accuracy = None        

    def get_model(self):
        return self._model

    def get_split_dataset(self):
        if self._split_dataset is None:
            raise ValueError("Split dataset is not available. Please train the model first.")
        return self._split_dataset

    def get_accuracy(self):
        if self._accuracy is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        return self._accuracy
    
    def set_parameters(self, parameters: dict) -> None:
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary.")
        self._parameters = parameters
        self._is_trained = False
        self._model = None
        self._accuracy = None

    def set_dataset(self, dataset: tuple, dataset_name: str) -> None:
        if len(dataset) != 2:
            raise ValueError("Dataset must be a tuple containing (X_train, y_train, X_test, y_test)")
        if not isinstance(dataset_name, str):
            raise ValueError("Dataset name must be a string.")
        if len(dataset_name) == 0:
            raise ValueError("Dataset name cannot be empty.")
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._split_dataset = None
        self._is_trained = False
        self._accuracy = None
    
    def train(self) -> None:
        if self._dataset is None:
            raise ValueError("Dataset not set. Please set a dataset first.")
        if self._parameters is None:
            raise ValueError("Parameters not set. Please set parameters first.")
        self._model, self._split_dataset = gen.grsf(self._dataset_name, self._parameters, debug=True)
        _, _, X, y = self._split_dataset
        if X is None or y is None:
            raise ValueError("Split dataset is not valid. Please check the dataset and parameters.")
        self._accuracy = gen.evaluate_grsf(self._model, X, y)
        self._is_trained = True
        if self._model is None:
            raise ValueError("Model training failed. Please check the parameters and dataset.")

    def __str__(self):
        if self._is_trained:
            return f"GRSF Model: {self._dataset_name}, Parameters: {self._parameters}, Accuracy: {self._accuracy:.2f} \n "
        else:
            return "GRSF Model: Not trained yet \n " 
        
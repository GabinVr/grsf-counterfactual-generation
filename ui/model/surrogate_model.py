import os
import sys

UI_ROOT = os.path.dirname(os.path.dirname(__file__))
if UI_ROOT not in sys.path:
    sys.path.append(UI_ROOT)
from utils import getDataset
from components.model_config import useDNN, dnnUtils
from typing import Dict
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import gen

class SurrogateModelObject:
    
    def __init__(self):
        self._model = None              
        self._model_arch = None
        # self._model_name = None         
        self._parameters = None         
        self._grsf_model = None  # trained grsf model        
        self._split_dataset = None            
        self._training_progress = ""    
        self._accuracy = None           
        self._epochs = 100              
        self._learning_rate = 0.001  
        self._training_progress_callback = None
        self._approximation_metrics = None

    def is_empty(self) -> bool:
        return self._model is None
    
    def has_changed(self, parameters: dict) -> bool:
        if self._parameters is None:
            return True

        return self._parameters != parameters
    
    def clear(self) -> None:
        self._model = None
        self._model_arch = None
        # self._model_name = None
        self._parameters = None
        self._grsf_model = None
        self._split_dataset = None
        self._training_progress = ""
        self._accuracy = None

    def is_trained(self) -> bool:
        """
        Check if the model has been trained.
        """
        return self._model is not None and self._accuracy is not None

    def get_model(self):
        if self._model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        return self._model
    
    def get_accuracy(self):
        if self._accuracy is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        return self._accuracy

    def get_approximation_metrics(self) -> Dict:
        if self._approximation_metrics is None:
            raise ValueError("Approximation metrics have not been evaluated yet. Please evaluate first.")
        return self._approximation_metrics
    
    def set_parameters(self, parameters: dict) -> None:
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary.")
        self._parameters = parameters

    def set_grsf_model(self, grsf_model) -> None:
        if grsf_model is None:
            raise ValueError("GRSF model cannot be None.")
        self._grsf_model = grsf_model

    def set_split_dataset(self, dataset: tuple) -> None:
        if not isinstance(dataset, tuple) or len(dataset) != 4:
            raise ValueError("Dataset must be containing (X_train, y_train, X_test, y_test).")
        self._split_dataset = dataset

    def set_training_parameters(self, epochs: int, learning_rate: float) -> None:
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive float.")
        self._epochs = epochs
        self._learning_rate = learning_rate
    
    def set_model(self, model:gen.BaseSurrogateClassifier) -> None:
        if model is None:
            raise ValueError("Model cannot be None.")
        setup = useDNN(model)
        self._model = setup.getSetupModel(**self._parameters)
        self._model_arch = self._model.get_architecture()
    
    def set_training_progress_callback(self, **kwargs):
        callback = kwargs.get('callback', None)
        if not callable(callback):
            raise ValueError("Training progress callback must be a callable function.")
        self._training_progress_callback = callback
    
    def _evaluate_approximation(self) -> Dict:
        """
        We evaluate how well the surrogate model approximates the GRSF model.
        returns a dictionary with :
            - 'agreement': float, percentage of agreement between surrogate and GRSF model predictions
            - 'surrogate_accuracy': float, accuracy of the surrogate model on the test set
        """
        if self._grsf_model is None:
            raise ValueError("GRSF model not set. Please set a GRSF model first.")
        if self._model is None:
            raise ValueError("Surrogate model not set. Please set a surrogate model first.")
        
        X_train, y_train, X_test, y_test = self._split_dataset
        if X_train is None or y_train is None:
            raise ValueError("Split dataset is not valid. Please check the dataset.")
        if X_test is None or y_test is None:
            raise ValueError("Test dataset is not valid. Please check the dataset.")
        
        surrogate_predictions = self._model.predict(X_test)
        grsf_predictions = self._grsf_model.predict(X_test)

        if isinstance(surrogate_predictions, torch.Tensor):
            surrogate_predictions = surrogate_predictions.cpu().numpy()
    
        agreement = (surrogate_predictions == grsf_predictions).mean() * 100

        self._approximation_metrics = {
            "agreement": agreement,
            "surrogate_accuracy": self._model.evaluate(X_test, y_test)
        }
        return self._approximation_metrics    

    def train(self) -> None:
        if self._split_dataset is None:
            raise ValueError("Dataset not set. Please set a dataset first.")
        if self._parameters is None:
            raise ValueError("Parameters not set. Please set parameters first.")
        if self._grsf_model is None:
            raise ValueError("GRSF model not set. Please set a GRSF model first.")

        X_train, y_train, X_test, y_test = self._split_dataset
        if X_train is None or y_train is None:
            raise ValueError("Split dataset is not valid. Please check the dataset.")
        self._model.train(
            X_train, y_train,
            epochs=self._epochs, 
            lr=self._learning_rate,
            training_callback=self._training_progress_callback,
            debug=True
        )

        self._accuracy = self._model.evaluate(X_test, y_test)
        self._evaluate_approximation()
        if self._accuracy is None:
            raise ValueError("Model training failed. Please check the parameters and dataset.")
        if self._approximation_metrics is None:
            raise ValueError("Approximation metrics not evaluated. Please evaluate the model first.")
        
    def get_info(self) -> dict:
        if self._model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        return {
            "model_architecture": self._model_arch,
            "parameters": self._parameters,
            "training_progress": self._training_progress,
            "accuracy": self._accuracy,
            "epochs": self._epochs,
            "learning_rate": self._learning_rate
        }
    
    def __str__(self):
        if self.is_empty():
            return "Surrogate model is not set."
        
        return (
            f"Surrogate Model Architecture: {str(self._model_arch)}\n"
            f"Parameters: {self._parameters}\n"
            f"Training Progress: {self._training_progress}\n"
            f"Accuracy: {self._accuracy}\n"
            f"Epochs: {self._epochs}, Learning Rate: {self._learning_rate}"
        )
"""
Author: Gabin Vrillault
mail: gabin[dot]vrillault[at]ecole[dot]ensicaen[dot]fr
Date: 2025-05-21

This module implements a Gradient-based time series classification algorithm
that aims to imitate the behavior of the Random Shapelet Forest (GRSF) classifier.
It also provides a gradient-based counterfactual generation method.

This module was written with the help of Claude 3.5 Sonnet and Cursor
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wildboar
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn import Linear, Module, ReLU, Sequential
from wildboar.datasets import load_synthetic_control
from wildboar.linear_model import RandomShapeletClassifier
from pyts.approximation import PiecewiseAggregateApproximation as PAA

##### UTILS #####

def getDatasetNames():
    """
    Get the names of the datasets available in wildboar.datasets compatible with GRSF.
    """
    return [dt for dt in wildboar.datasets.list_datasets() if dt not in ["OliveOil", 
                                                                        "Phoeme", 
                                                                        "PigAirwayPressure",
                                                                        "PigArtPressure",
                                                                        "PigCVP",
                                                                        "Fungi",
                                                                        "FiftyWords"]]
def getDataset(dataset_name:str):
    return pd.DataFrame(wildboar.datasets.load_dataset(dataset_name)[0])

def _generate_zero_padded_samples(X, nb_samples):
    """
    Generate zero-padded samples from the input time series.
    This method generates `nb_samples` samples by zero-padding the input time series.
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy.ndarray or a torch.Tensor")
    
    zero_padded_samples = np.zeros((nb_samples, X.shape[0]))
    
    start = 0
    step = X.shape[0] // nb_samples
    for i, sample in enumerate(zero_padded_samples):
        # We take a chunk of the time series and zero-pad it
        end = start + step
        if end > X.shape[0]:
            end = X.shape[0]
        sample = np.insert(sample, start, X[start:end], axis=0)
        # We need to trim the sample to the right size
        sample = sample[:X.shape[0]]
        zero_padded_samples[i] = sample
        start += step
    
    return zero_padded_samples

def preprocess_dataset(X, y):
    """
    Preprocess the dataset:
    - Shift all labels to start from 0
    - For example:
        - [-1, 0, 1] -> [0, 1, 2]
        - [-1, 1] -> [0, 1]
        - [0, 1, 2] -> [0, 1, 2] (unchanged)
    """
    unique_labels = np.unique(y)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    return X, y


def grsf(dataset:str, params:dict, debug:bool=True):
    """
    Function to create a GRSF classifier and fit it to the dataset.
    
    Parameters:
    - dataset: str, name of the dataset to be used
    must be one of the datasets available in wildboar.datasets
    - params: dict, parameters for the GRSF algorithm
    must contain the following keys:
        - n_shapelets: int, number of shapelets to be used
        - metric: str, distance metric to be used
        - min_shapelet_size: int, minimum size of the shapelets
        - max_shapelet_size: int, maximum size of the shapelets
        - alphas: list, list of alpha values to be used
    return classifier: RandomShapeletClassifier, (X_train, y_train, X_test, y_test) tuple of the split dataset
    """
    try:
        X, y = wildboar.datasets.load_dataset(dataset)
        # We need to preprocess the dataset to "clean" it
        X, y = preprocess_dataset(X, y)
    except ValueError as e:
        print(f"Error loading dataset {dataset}: {e}")
        return (1, "Error-code: Dataset not found or invalid format")
    
    try:
        classifier = RandomShapeletClassifier(
            n_shapelets=params["n_shapelets"],
            metric=params["metric"],
            min_shapelet_size=params["min_shapelet_size"],
            max_shapelet_size=params["max_shapelet_size"],
            alphas=params["alphas"]
        )
    except ValueError as e:
        print(f"Error creating classifier: {e}")
        return (1, "Error-code: Invalid parameters for RandomShapeletClassifier")
    
    # Let's split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Fit the classifier
    try:
        classifier.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error fitting classifier: {e}")
        return (1, "Error-code: Failed to fit the classifier")
    
    return classifier, (X_train, y_train, X_test, y_test)

def evaluate_grsf(classifier:RandomShapeletClassifier, X_test, y_test, debug:bool=True):
    """
    Evaluate the GRSF classifier on the test set.
    
    Parameters:
    - classifier: RandomShapeletClassifier, the trained classifier
    - X_test: np.ndarray, test set features
    - y_test: np.ndarray, test set labels
    - debug: bool, whether to print debug information
    
    Returns:
    - accuracy: float, accuracy of the classifier on the test set
    """
    assert isinstance(classifier, RandomShapeletClassifier), "classifier must be a RandomShapeletClassifier"
    
    try:
        accuracy = classifier.score(X_test, y_test)
        if debug:
            print(f"Accuracy: {accuracy:.2f}")
        return accuracy
    except ValueError as e:
        print(f"Error evaluating classifier: {e}")
        return None

def plot_shapelets(clf, dataset:str):
    """
    Function to plot the shapelets learned by the classifier.
    
    Parameters:
    - clf: RandomShapeletClassifier, the trained classifier
    - dataset: str, name of the dataset to be used
    """
    assert isinstance(clf, RandomShapeletClassifier), "clf must be a RandomShapeletClassifier"
    assert isinstance(dataset, str), "dataset must be a string"
    assert dataset in wildboar.datasets.list_datasets(), f"dataset {dataset} not found in wildboar.datasets"
    try:
        # Get the shapelet transform step from the pipeline
        shapelet_transform = clf.pipe_.named_steps['transform']
        shapelets = shapelet_transform.embedding_.attributes

        for i, shapelet in enumerate(shapelets):
            shapelet = shapelet[1][1]
            plt.figure(figsize=(10, 5))
            plt.plot(shapelet, label=f"Shapelet {i+1}", color="blue", linewidth=2)
            plt.title(f"Shapelet {i+1} from {dataset}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid()
            plt.savefig(f"shapelet_{i+1}_{dataset}.png")
            plt.close()

    except Exception as e:
        print(f"Error plotting shapelets: {e}")
        return

class BaseSurrogateClassifier(Module):
    """
    Base class for surrogate classifiers.
    This class enables the use of multiple surrogate classifiers.
    (different architectures)
    """
    def __init__(self):
        super(BaseSurrogateClassifier, self).__init__()
    
    def forward(self, x):
        """
        Forward pass for the surrogate classifier.
        """
        raise NotImplementedError("Forward must be implemented in the subclass")
 
    def predict(self, x):
        """
        Predict the class labels for the input data.
        """
        with torch.no_grad():
            x = self.forward(x)
            _, predicted = torch.max(x, 1)
        return predicted

    def train(self,
              X_train, 
              y_train, 
              epochs:int=100, 
              lr:float=0.01, 
              criterion:torch.nn.Module=torch.nn.CrossEntropyLoss(), 
              optimizer:torch.optim.Optimizer=torch.optim.Adam,
              debug:bool=True,
              training_callback:callable=None
              ):
        """
        Train the surrogate classifier.
        X_train is a torch.Tensor of shape (n_samples, n_features)
        y_train is a torch.Tensor of shape (n_samples,)
        epochs is the number of epochs to train the model
        lr is the learning rate
        criterion is the loss function
        optimizer is the optimizer
        debug is a boolean to print the training progress
        training_callback is a function that will be called at the end of each epoch
        """
        # self.parameters() is a generator that yields the parameters of the model
        optimizer = optimizer(self.parameters(), lr=lr)
        # if the dataset is not a torch.Tensor, we convert it to a torch.Tensor
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        # Check if the input data is 2D, if not, reshape it
        if len(X_train.shape) == 1:
            X_train = X_train.unsqueeze(0)
        elif len(X_train.shape) == 2:
            X_train = X_train.unsqueeze(1)
        # Check if the labels are 1D, if not, reshape them
        if len(y_train.shape) > 1:
            y_train = y_train.squeeze()
        # Check if the labels are of type long (int64)
        if y_train.dtype != torch.long:
            y_train = y_train.long()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0 and debug:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            if training_callback is not None:
                training_callback(epoch, loss.item())
        if debug:
            print("Training complete.")
        return self
    
    def evaluate(self, X_test, y_test, debug:bool=True):
        """
        Evaluate the surrogate classifier.
        X_test is a torch.Tensor of shape (n_samples, n_features)
        y_test is a torch.Tensor of shape (n_samples,)
        debug is a boolean to print the evaluation progress
        """
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test, dtype=torch.long)
        # Check if the input data is 2D, if not, reshape it
        if len(X_test.shape) == 1:
            X_test = X_test.unsqueeze(0)
        elif len(X_test.shape) == 2:
            X_test = X_test.unsqueeze(1)
        # Check if the labels are 1D, if not, reshape them
        if len(y_test.shape) > 1:
            y_test = y_test.squeeze()
        with torch.no_grad():
            outputs = self.forward(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
        if debug:
            print(f"Accuracy: {accuracy:.2f}")
        return accuracy
    
    def get_params(self):
        """
        Return model parameters.
        This method should be implemented in the subclass to return the parameters of the model.
        """
        raise NotImplementedError("get_params must be implemented in the subclass")
   
class SurrogateContext:
    """
    Class for creating a surrogate context for the GRSF classifier.
    This class enables the use of multiple surrogate classifiers.
    """
    def __init__(self, classifier: BaseSurrogateClassifier):
        self._classifier = classifier
    
    def set_classifier(self, classifier: BaseSurrogateClassifier):
        """
        Set the classifier for the surrogate context.
        """
        self._classifier = classifier
        return self
    
    def train(self,
              X_train,
              y_train,
              epochs:int=100,
              lr:float=0.01,
              criterion:torch.nn.Module=torch.nn.CrossEntropyLoss(),
              optimizer:torch.optim.Optimizer=torch.optim.Adam,
              debug:bool=True):
        """
        Train the classifier.
        X_train is a torch.Tensor of shape (n_samples, n_features)
        y_train is a torch.Tensor of shape (n_samples,)
        epochs is the number of epochs to train the model
        lr is the learning rate
        criterion is the loss function
        optimizer is the optimizer
        debug is a boolean to print the training progress
        """
        self._check_classifier()
        return self._classifier.train(X_train, y_train, epochs=epochs, lr=lr, criterion=criterion, optimizer=optimizer, debug=debug)
    
    def evaluate(self, X_test, y_test, debug:bool=True):
        """
        Evaluate the classifier.
        """
        self._check_classifier()
        return self._classifier.evaluate(X_test, y_test, debug=debug)

    def predict(self, x):
        """
        Predict the class labels for the input data.
        """
        self._check_classifier()
        return self._classifier.predict(x)
    
    def forward(self, x):
        """
        Forward pass for the surrogate classifier.
        """
        self._check_classifier()
        return self._classifier.forward(x)

    def get_params(self):
        """
        Return model parameters.
        This method should be implemented in the subclass to return the parameters of the model.
        """
        self._check_classifier()
        return self._classifier.get_params()

    def _check_classifier(self):
        """
        Check if the classifier is set.
        """
        if self._classifier is None:
            raise ValueError("Classifier not set")
    
    def __call__(self, x):
        """
        Call the classifier without the softmax layer.
        """
        self._check_classifier()
        return self._classifier.forward(x)
 
    # Forward pass is not needed in the surrogate context

class SimpleSurogateClassifier(BaseSurrogateClassifier):
    """
    A surrogate classifier for the GRSF algorithm.
    This is a simple feedforward neural network with one hidden layer.
    """
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(SimpleSurogateClassifier, self).__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CounterFactualCrafting:
    """
    Class for crafting counterfactuals for the GRSF algorithm using a surrogate classifier.
    """
    def __init__(self, grsf_classifier:RandomShapeletClassifier, surrogate_classifier:BaseSurrogateClassifier):
        self.grsf_classifier = grsf_classifier
        self.surrogate_classifier = surrogate_classifier
        self.closest_to_boundary = None
        print(f"DEBUG gen.py CounterFactualCrafting: {grsf_classifier}, {surrogate_classifier}")

    def loss_fn(self, x, target, base, base_label):
        """
        Loss function for the counterfactual crafting.
        x is a torch.Tensor that is the counterfactual candidate
        target is a torch.Tensor that is the target sample
        base is a torch.Tensor that is the base sample
        base_label is the label of the base sample
        """
        loss = torch.norm(self.surrogate_classifier(x) - self.surrogate_classifier(target)).pow(2)
        # loss += self.beta * torch.norm(x - base).pow(2)
        return loss
    
    def generate_counterfactual(self, target, base, base_label, 
                                lr:float=0.01,
                                epochs:int=100,
                                debug:bool=True,
                                beta:float=0.5,
                                training_callback:callable=None
                                ):
        """
        Generate counterfactuals 
        :target: torch.Tensor, the target sample
        :base: torch.Tensor, the base sample
        :base_label: int, the label of the base sample
        :lr: float, learning rate for the optimization
        :epochs: int, number of epochs for the optimization
        :debug: bool, whether to print debug information
        :beta: float, regularization parameter
        :training_callback: callable, a function to call at the end of each epoch for custom training logic
        Generates a counterfactual sample that is close to the base sample
        but has a different prediction from the GRSF classifier.
        """
        # Verify that inputs are torch tensors
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        if not isinstance(base, torch.Tensor):
            base = torch.tensor(base, dtype=torch.float32)
        if not isinstance(base_label, torch.Tensor):
            base_label = torch.tensor(base_label, dtype=torch.long)

        
        x = base.clone().detach().requires_grad_(True)

        for epoch in range(epochs):
            x_backup = x.clone().detach().numpy()
            # Forward step
            loss = self.loss_fn(x, target, base, base_label)
            loss.backward()
            
            # Check if grad exists
            if x.grad is None:
                x.grad = torch.zeros_like(x)
        
            x_temp = x - lr * x.grad

            # Backward step
            x = (x_temp + lr * beta * base) / (1 + lr * beta)
            
            x = x.clone().detach().requires_grad_(True)
            x_test = x.clone().detach().numpy()

            # Check if the counterfactual as changed prediction
            pred = self.grsf_classifier.predict(x_test.reshape(1, -1))
            pred_backup = self.grsf_classifier.predict(x_backup.reshape(1, -1))

            if pred != pred_backup:
                if debug:
                    print(f"Counterfactual has changed prediction at epoch {epoch+1}")
                self.closest_to_boundary = x_backup
                break
        
            if (epoch+1) % 10 == 0 and debug:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            if training_callback is not None:
                training_callback(epoch, loss.item())
    
        if debug:
            print("Counterfactual generation complete.")
        return x
    
    def generate_local_counterfactuals(self, target, base, base_label, binary_mask, 
                                       lr:float=0.01,
                                       epochs:int=100,
                                       debug:bool=True,
                                       beta:float=0.5,
                                       training_callback:callable=None):
        """
        Generate local counterfactuals
        :target: torch.Tensor, the target sample
        :base: torch.Tensor, the base sample
        :base_label: int, the label of the base sample
        :binary_mask: torch.Tensor, a binary mask indicating the regions to be modified (1) or frozen (0)
        :lr: float, learning rate for the optimization
        :epochs: int, number of epochs for the optimization
        :debug: bool, whether to print debug information
        :beta: float, regularization parameter
        :training_callback: callable, a function to call at the end of each epoch for custom training logic
        """
        print(f"DEBUG gen.py generate_local_counterfactuals: {target.shape}, {base.shape}, {base_label}, {binary_mask.shape}")
        assert isinstance(binary_mask, torch.Tensor), "binary_mask must be a torch.Tensor"
        assert 1 in binary_mask.unique(), "binary_mask must contain at least one 1"
        assert 0 in binary_mask.unique(), "binary_mask must contain at least one 0"
        assert binary_mask.shape == base.shape, "binary_mask must have the same shape as base"

        # Clone base as starting point
        x = base.clone().detach().requires_grad_(True)
        
        modify_mask = binary_mask.clone()  # Regions to modify
        freeze_mask = 1 - binary_mask      # Regions to freeze
        
        for epoch in range(epochs):
            x_backup = x.clone().detach().numpy()

            # Forward pass
            loss = self.loss_fn(x, target, base, base_label)
            
            # Add regularization to keep frozen regions unchanged
            frozen_penalty = torch.norm((x - base) * freeze_mask).pow(2)
            total_loss = loss + 10.0 * beta * frozen_penalty
            
            total_loss.backward()
            
            # Check if grad exists
            if x.grad is None:
                x.grad = torch.zeros_like(x)
            
            # Apply mask to gradient: only modify allowed regions
            x.grad = x.grad * modify_mask
            
            if debug and (epoch % 50 == 0):
                print(f"Epoch {epoch}:")

            # Update step
            x_temp = x - lr * x.grad
            
            # Apply regularization in parameter space
            x = (x_temp + lr * beta * base) / (1 + lr * beta)
            
            # CRUCIAL: Force frozen regions to stay exactly as base
            x = x * modify_mask + base * freeze_mask
            
            # Prepare for next iteration
            x = x.clone().detach().requires_grad_(True)
            x_test = x.clone().detach().numpy()

            # Check if prediction changed
            pred = self.grsf_classifier.predict(x_test.reshape(1, -1))
            pred_backup = self.grsf_classifier.predict(x_backup.reshape(1, -1))

            if pred != pred_backup:
                if debug:
                    print(f"Counterfactual has changed prediction at epoch {epoch+1}")
                    print(f"  From class {pred_backup} to class {pred}")
                self.closest_to_boundary = x_backup
                break
                
            if training_callback is not None:
                training_callback(epoch, total_loss.item())
        
        if debug:
            print("Counterfactual generation complete.")
        print(f"DEBUG gen.py generate_local_counterfactuals: Final counterfactual shape: {x.shape}, head: {x[:10]}")
        return x

class LocalCounterFactualCrafting:
    """
    Class for crafting locally based counterfactuals for the GRSF algorithm using a surrogate classifier.
    Essentially here we use CounterFactual crafting on smaller sections of time series
    NOTE: Probably still have to rewrite the function to generate the counterfactuals
    """
    def __init__(self, grsf_classifier:RandomShapeletClassifier, surrogate_classifier:BaseSurrogateClassifier, beta:float=0.5):
        self.grsf_classifier = grsf_classifier
        self.surrogate_classifier = surrogate_classifier
        self.closest_to_boundary = None
        self.beta = beta
    
    def local_PAA(self, target, base, base_label, lr:float=0.01, epochs:int=100, debug:bool=True, size_percentage:float=0.5):
        """
        Uses PAA methods to split the time series.
        1. We clone the base as X 
        2. We create Xi splits, with zero padding around sample part of X
        3. We generate counterfactuals for each Xi using Global method if possible
        4. We retrieve a list of local counterfactuals
        """
        if isinstance(base, torch.Tensor):
            X = base.clone().detach().requires_grad_(True)
        elif isinstance(base, np.ndarray):
            X = torch.tensor(base, dtype=torch.float32).clone().detach().requires_grad_(True)
        else:
            raise ValueError("base must be a torch.Tensor or a numpy.ndarray")

        paaTransformer = PAA(window_size=None, output_size=size_percentage, overlapping=True)

        paaSections = paaTransformer.transform(X.detach().numpy().reshape(1, -1))
        print(f"X : {X.shape}, head: {X[:10]}")
        print(f"paaSections shape: {paaSections.shape}, head: {paaSections[0][:10]}")



    # def local_APCA():
    #     """
    #     Uses APCA methods to split the time series.
    #     """
    #     pass

###### TESTS ######

def test_generate_zero_padded_samples():
    """
    Test the generate_zero_padded_samples function.
    """
    dataset = wildboar.datasets.load_dataset("ECG200")
    X, y = dataset
    nb_samples = 5
    sample = int(np.random.rand(X.shape[1])[0])
    zero_padded_samples = _generate_zero_padded_samples(X[sample], nb_samples)
    assert zero_padded_samples.shape == (nb_samples, (X[sample].shape)[0]), "Zero padded samples shape mismatch"
    assert np.all(np.isfinite(zero_padded_samples)), "Zero padded samples contain NaN or Inf values"
    start = 0
    step = (X[sample].shape)[0] // nb_samples
    for i in range(nb_samples):
        assert np.count_nonzero(zero_padded_samples[i]) > 0, f"Sample {i} is empty"
        assert np.count_nonzero(zero_padded_samples[i]) <= (X[sample].shape)[0], f"Sample {i} is too long"
        assert np.all(zero_padded_samples[i][:start] == 0), f"Sample {i} has non-zero values before start"
        end = start + step
        if end > (X[sample].shape)[0]:
            end = (X[sample].shape)[0]
        assert np.all(zero_padded_samples[i][start:end] == X[sample][start:end]), f"Sample {i} has wrong values in the middle"
        start += step
    assert np.all(zero_padded_samples[-1][start:] == 0), "Last sample has non-zero values after end"
    print("Test passed: generate_zero_padded_samples")

if __name__ == "__main__":
    
    print(f"Datasets: {wildboar.datasets.list_datasets()}")
    print(f"Repositories: {wildboar.datasets.list_repositories()}")

    swedish_leaf = wildboar.datasets.load_dataset("SwedishLeaf")
    print(f"Swedish Leaf: {type(swedish_leaf)}, {len(swedish_leaf)}")
    print(f" {swedish_leaf[0].shape}, {swedish_leaf[1].shape}")
    print(f" {swedish_leaf[0][0].shape}, {swedish_leaf[0][1].shape}, {swedish_leaf[1][0]}")
    # let's plot the first sample
    plt.figure(figsize=(10, 5))
    plt.plot(swedish_leaf[0][0], label="Sample 1", color="blue", linewidth=2)
    plt.plot(swedish_leaf[0][1], label="Sample 2", color="red", linewidth=2)
    plt.title("Swedish Leaf Sample")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig("swedish_leaf_sample.png")

    ecg200 = wildboar.datasets.load_dataset("ECG200")
    f = lambda x: {-1:("Normal", "red" ), 1:("Abnormal", "blue")}[int(x)]
    print(f"ECG200: {type(ecg200)}, {len(ecg200)}")
    print(f" {ecg200[0].shape}, {ecg200[1].shape}") 
    print(f" {ecg200[0][0].shape}, {ecg200[0][1].shape}, {ecg200[1][0]}")
    # let's plot the first sample
    plt.figure(figsize=(10, 5))
    label, color = f(ecg200[1][0])
    plt.plot(ecg200[0][0], label=label, color=color, linewidth=2)
    label, color = f(ecg200[1][1])
    plt.plot(ecg200[0][1], label=label, color=color, linewidth=2)
    plt.title("ECG200 Sample")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig("ecg200_sample.png")

    cinCECGT = wildboar.datasets.load_dataset("CinCECGTorso")
    print(f"CinCECGT: {type(cinCECGT)}, {len(cinCECGT)}")
    print(f" {cinCECGT[0].shape}, {cinCECGT[1].shape}") # (1420, 1639), (1420,)
    print(f" {cinCECGT[0][0].shape}, {cinCECGT[0][1].shape}, {cinCECGT[1][0]}")
    # let's plot the first sample
    plt.figure(figsize=(10, 5))
    f = lambda x: {1:("person_1", "red" ), 2:("person_2", "blue"), 3:("person_3", "green"), 4:("person_4", "orange")}[int(x)]
    label, color = f(cinCECGT[1][0])
    print(f"label: {label}")
    plt.plot(cinCECGT[0][0], label=label, color=color, linewidth=2)
    label, color = f(cinCECGT[1][1])
    print(f"label: {label}")
    plt.plot(cinCECGT[0][1], label=label, color=color, linewidth=2)
    label, color = f(cinCECGT[1][2])
    print(f"label: {label}")
    plt.plot(cinCECGT[0][2], label=label, color=color, linewidth=2)
    label, color = f(cinCECGT[1][6])
    print(f"label: {label}")
    plt.plot(cinCECGT[0][6], label=label, color=color, linewidth=2)
    plt.title("CinCECGT Sample")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig("cinCECGT_sample.png")



    # let's run the GRSF algorithm on the ECG200 dataset
    params = {
        "n_shapelets": 100,
        "metric": "euclidean",
        "min_shapelet_size": 0.0,
        "max_shapelet_size": 1.0,
        "alphas": [0.1, 0.5, 0.9]
    }
    clf,_ = grsf("ECG200", params)

    dataset = wildboar.datasets.load_dataset("ECG200")
    surrogate = SimpleSurogateClassifier(input_size=dataset[0].shape[1], hidden_size=100, output_size=len(np.unique(dataset[1])))
    dnn_classifier = SurrogateContext(surrogate)
    X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=42)
    print(f"X_test 0: {y_test[0]}, X_test 1: {y_test[3]}")

    # Convert the negative class labels to positive
    y_train = np.where(y_train == -1, 0, y_train)
    y_test = np.where(y_test == -1, 0, y_test)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    # Train the surrogate classifier
    dnn_classifier.train(X_train, y_train, epochs=100, lr=0.01)
    # Evaluate the surrogate classifier
    dnn_classifier.evaluate(X_test, y_test)

    # Generate counterfactual
    target = X_test[0]
    base = X_test[3]
    base_label = y_test[3]
    print(f"Setup of the counterfactual crafting:")
    print(f"Target: {target}, Base: {base}, Base label: {base_label}, Target label: {y_test[0]}")
    counterfactual_crafter = CounterFactualCrafting(clf, dnn_classifier, beta=0.5)
    counterfactual = counterfactual_crafter.generate_counterfactual(target, base, base_label, lr=0.01, epochs=100)
    print(f"Counterfactual: {counterfactual}")
    # Let's convert the counterfactual back to a time series
    counterfactual = counterfactual.detach().numpy()
    print(f"Counterfactual shape: {counterfactual.shape}")
    # Let's plot the counterfactual
    plt.figure(figsize=(10, 5))
    plt.plot(counterfactual, label="Counterfactual", color="blue", linewidth=2)
    plt.title("Counterfactual")
    original = base.detach().numpy()
    plt.plot(original, label="Original", color="red", linewidth=2)
    target = target.detach().numpy()
    plt.plot(target, label="Target", color="green", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig("counterfactual.png")

    pred = clf.predict(counterfactual.reshape(1, -1))
    print(f"Predicted label: {pred}, original label: {base_label}")

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import gen 

import numpy as np
import wildboar.datasets as wb_datasets
import wildboar.distance as wb_distance
import torch
import torch.nn as nn
from pyts.approximation import PiecewiseAggregateApproximation as PAA

# Utils
def get_base_target_pair(X_test, y_test, nb_samples:int):
    """
    Get a list of base and target samples
    """
    base_target_pairs = []
    classes = np.unique(y_test)
    
    # Group samples by class
    class_samples = {c: X_test[y_test == c] for c in classes}
    
    # Sample from each class
    for base_class in classes:
        if len(class_samples[base_class]) < nb_samples:
            print(f"Not enough samples for class {base_class}")
            return None
            
        for i in range(nb_samples):
            # Get target class different from base class
            target_class = np.random.choice(classes[classes != base_class])
            target_sample = class_samples[target_class][np.random.randint(len(class_samples[target_class]))]
            base_target_pairs.append((
                (class_samples[base_class][i], base_class),
                (target_sample, target_class)
            ))
            
    return base_target_pairs

def counterfactual_batch(dataset:str, params:dict, nb_samples:int=10, nn_classifier=None, debug:bool=True):
    """
    Generate a batch of nb_samples counterfactuals for a given dataset
    dataset: the dataset to use (cf wildboar.datasets)
    params: parameters for the GRSF classifier
    nb_samples: the number of counterfactuals to generate
    nn_classifier: a pre-trained surrogate classifier
    return: a list of triplets (counterfactual, target, base)
    """
    # First we need to train the GRSF classifier
    grsf_classifier, data = gen.grsf(dataset, params, debug=False)
    X_train, y_train, X_test, y_test = data

    # Then we need to create a surrogate classifier
    if not nn_classifier:
        nn_classifier = gen.SimpleSurogateClassifier(input_size=X_train.shape[1], hidden_size=100, output_size=len(np.unique(y_train)))
    
    surrogate = gen.SurrogateContext(nn_classifier)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Train the surrogate classifier
    surrogate.train(X_train, y_train, debug=debug)

    # Test the surrogate classifier
    surrogate.evaluate(X_test, y_test, debug=True)

    # Create a counterfactual crafting object
    counterfactual = gen.CounterFactualCrafting(grsf_classifier, surrogate)

    # Get a batch of base and target pairs
    base_target_pairs = get_base_target_pair(X_test, y_test, nb_samples)
    if base_target_pairs is None:
        print("Not enough samples for counterfactual crafting")
        return None
    counterfactuals = []
    for base, target in base_target_pairs:
        (base_sample, base_label) = base
        (target_sample, _) = target
        counterfactual_sample = counterfactual.generate_counterfactual(target_sample, base_sample, base_label, debug=debug)
        counterfactuals.append((counterfactual_sample, target, base))
    
    return counterfactuals
        
def evaluate_counterfactuals(counterfactuals:list):
    """
    Evaluate the quality of the counterfactuals
    """
    # Check if the counterfactuals are valid
    for counterfactual, target, base in counterfactuals:
        assert counterfactual.shape == base[0].shape
        assert counterfactual.shape == target[0].shape
    
    # Compute distance between counterfactual and base
    base_distances = []
    for counterfactual, target, base in counterfactuals:
        counterfactual = counterfactual.detach().numpy()
        base = base[0].detach().numpy()
        base_distances.append(np.linalg.norm(counterfactual - base))
    
    # Compute distance between counterfactual and target
    target_distances = []
    for counterfactual, target, base in counterfactuals:
        counterfactual = counterfactual.detach().numpy()
        target = target[0].detach().numpy()
        target_distances.append(np.linalg.norm(counterfactual - target))
    

    return base_distances, target_distances


#### Tests

def test_get_base_target_pair():
    datasets = wb_datasets.list_datasets()
    fails = 0
    for dataset in datasets:
        print(f"Debug: dataset: {dataset}")
        X, y = wb_datasets.load_dataset(dataset)
        base_target_pairs = get_base_target_pair(X, y, 10)
        if base_target_pairs is None:
            print("test_get_base_target_pair: failed")
            fails += 1
            continue
        for base, target in base_target_pairs:
            assert base[1] != target[1]
            assert base[0].shape == target[0].shape
    
        print("test_get_base_target_pair: base target pairs are valid")
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        base_target_pairs = get_base_target_pair(X, y, 10)
        if base_target_pairs is None:
            print("test_get_base_target_pair: failed")
            fails += 1
            continue
        for base, target in base_target_pairs:
            assert base[1] != target[1]
            assert base[0].shape == target[0].shape
            assert isinstance(base[0], torch.Tensor)
            assert isinstance(target[0], torch.Tensor)
        print("test_get_base_target_pair: base target pairs torch tensors are valid")
    
    print(f"test_get_base_target_pair: passed with {fails} fails")

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

def distance_based_analysis(grsf_classifier, counterfactuals:list, n_segments:int=10):
    """
    Computes and prints distances based informations about counterfactuals
    """
    # Check if the counterfactuals are valid
    for counterfactual, target, base in counterfactuals:
        assert counterfactual.shape == base[0].shape
        assert counterfactual.shape == target[0].shape
    


    # Compute distance between counterfactual and base
    base_distances_euclid = []
    base_distances_dtw = []
    nb_valid_counterfactuals = 0
    for counterfactual, target, base in counterfactuals:
        counterfactual = counterfactual.detach().numpy()
        base = base[0].detach().numpy()
        base_distances_euclid.append(np.linalg.norm(counterfactual - base))
        base_distances_dtw.append(wb_distance.dtw.dtw_distance(counterfactual, base, r=1.0))
        if grsf_classifier.predict(counterfactual.reshape(1, -1)) == grsf_classifier.predict(base.reshape(1, -1)):
            nb_valid_counterfactuals += 1
    base_sparcity = np.sum(counterfactual != base)/ counterfactual.size * 100
    
    print(f"Counter - Base | Euclid {np.mean(base_distances_euclid)}, DTW {np.mean(base_distances_dtw)}, sparcity {base_sparcity}")
    
    # Compute distance between counterfactual and target
    target_distances_euclid = []
    target_distances_dtw = []
    for counterfactual, target, base in counterfactuals:
        counterfactual = counterfactual.detach().numpy()
        target = target[0].detach().numpy()
        target_distances_euclid.append(np.linalg.norm(counterfactual - target))
        target_distances_dtw.append(wb_distance.dtw.dtw_distance(counterfactual, target, r=1.0))
    target_sparcity = np.sum(counterfactual != target) / counterfactual.size * 100
    print(f"Counter - Target | Euclid {np.mean(target_distances_euclid)}, DTW {np.mean(target_distances_dtw)}, sparcity {target_sparcity}")

    # Let's apply PAA to the counterfactuals, base and target
    paa = PAA(window_size=n_segments)
    paa_base = []
    paa_target = []
    for counterfactual, target, base in counterfactuals:
        cf_paa = paa.transform(counterfactual.detach().numpy().reshape(1, -1))[0]
        target_paa = paa.transform(target[0].detach().numpy().reshape(1, -1))[0]
        base_paa = paa.transform(base[0].detach().numpy().reshape(1, -1))[0]
        paa_base.append(np.linalg.norm(cf_paa - base_paa))
        paa_target.append(np.linalg.norm(cf_paa - target_paa))
    print(f"PAA Counter - Base | Euclid {np.mean(paa_base)}, DTW {np.mean(wb_distance.dtw.dtw_distance(paa_base, paa_target, r=1.0))}")
    print(f"PAA Counter - Target | Euclid {np.mean(paa_target)}, DTW {np.mean(wb_distance.dtw.dtw_distance(paa_target, paa_base, r=1.0))}")
    # Print the number of valid counterfactuals
    print(f"Valid counterfactuals: {nb_valid_counterfactuals} out of {len(counterfactuals)} ({nb_valid_counterfactuals/len(counterfactuals)*100:.2f}%)")
    


if __name__ == "__main__":
    # test_get_base_target_pair()
    # Does not work with the following datasets:
    # OliveOil
    # Phoeme
    # PigAirwayPressure
    # PigArtPressure
    # PigCVP
    # Fungi
    # FiftyWords

    print(f"Counterfactual crafting:")

    dataset_name = "ECG200"
    params = {
        "n_shapelets": 10,
        "metric": "euclidean",
        "min_shapelet_size": 0.0,
        "max_shapelet_size": 1.0,
        "alphas": [0.1, 0.5, 0.9]
    }
    dataset = wb_datasets.load_dataset(dataset_name)
    grsf_classifier, _ = gen.grsf(dataset_name, params, debug=False)
    print(f"Simple classifier:")
    classifier = gen.SimpleSurogateClassifier(input_size=dataset[0].shape[1], hidden_size=100, output_size=len(np.unique(dataset[1])))
    print(f"Dataset: {dataset_name} with classes: {np.unique(dataset[1])}")
    counterfactuals = counterfactual_batch(dataset_name, params, 10, classifier, debug=False)
    distance_based_analysis(grsf_classifier, counterfactuals)

    print(f"LSTM classifier:")
    classifier = LSTMClassifier(input_size=dataset[0].shape[1], hidden_size=100, num_layers=2, output_size=len(np.unique(dataset[1])))
    counterfactuals = counterfactual_batch(dataset_name, params, 10, classifier, debug=False)
    distance_based_analysis(grsf_classifier, counterfactuals)
    print(f"CNN classifier:")
    classifier = CNNClassifier(input_size=dataset[0].shape[1], hidden_size=100, output_size=len(np.unique(dataset[1])))
    counterfactuals = counterfactual_batch(dataset_name, params, 10, classifier, debug=False)
    distance_based_analysis(grsf_classifier, counterfactuals)

    input("Press Enter to continue with local counterfactuals crafting...")
    # Let's test local counterfactuals crafting


    print(f"Local counterfactual crafting:")
    grsf_classifier, data = gen.grsf(dataset_name, params, debug=False)
    X_train, y_train, X_test, y_test = data
    classifier = gen.SimpleSurogateClassifier(input_size=X_train.shape[1], hidden_size=100, output_size=len(np.unique(y_train)))
    crafter = gen.CounterFactualCrafting(grsf_classifier, classifier, beta=0.1)

    target = X_test[0]
    base = X_test[1]
    base_label = y_test[1]

    target = torch.tensor(target, dtype=torch.float32)
    base = torch.tensor(base, dtype=torch.float32)
    base_label = torch.tensor(base_label, dtype=torch.long)

    counterfactual_samples = crafter.generate_local_counterfactuals(target, base, base_label, nb_samples=5, debug=True)
    print(f"Counterfactual samples: {len(counterfactual_samples)}")

    # Plotting the counterfactuals compared to the base and target

    import matplotlib.pyplot as plt
    import os
    size = len(counterfactual_samples)
    # Matrix of plots with superposed counterfactuals, base and target
    # The goal is to have a grid of plots to visualize the counterfactuals
    # compared to the base and target

    # Create subplot grid
    cols = 3
    rows = (size + cols - 1) // cols  # Ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Flatten axes array for easier indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Convert tensors to numpy for plotting
    target_np = target.detach().numpy()
    base_np = base.detach().numpy()
    
    for i, counterfactual in enumerate(counterfactual_samples):
        if i >= len(axes):
            break
            
        counterfactual_np = counterfactual.detach().numpy()
        
        axes[i].plot(base_np, label='Base', color='blue', alpha=0.7, linestyle='dashed')
        axes[i].plot(target_np, label='Target', color='red', alpha=0.7)
        axes[i].plot(counterfactual_np, label='Counterfactual', color='green', alpha=0.8, linestyle='dashdot')
        axes[i].set_title(f'Counterfactual {i+1}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(size, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    i = 0
    while os.path.exists(f"local_counterfactual_{i}.png"):
        i += 1
    # Let's save the plot
    plt.savefig(f"local_counterfactual_{i}.png")
    plt.show()
        

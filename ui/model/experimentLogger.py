import time 
import json
import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy arrays and other non-serializable objects."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # For custom objects, try to serialize their __dict__
            return str(obj)
        return super().default(obj)


def serialize_data(data):
    """
    Recursively serialize data to make it JSON-compatible.
    Handles NumPy arrays, pandas objects, and other complex types.
    """
    if data is None:
        return None
    elif isinstance(data, dict):
        return {str(key): serialize_data(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [serialize_data(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float16, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, pd.DataFrame):
        return {
            'type': 'DataFrame',
            'data': data.to_dict('records'),
            'columns': data.columns.tolist(),
            'index': data.index.tolist()
        }
    elif isinstance(data, pd.Series):
        return {
            'type': 'Series',
            'data': data.tolist(),
            'index': data.index.tolist(),
            'name': data.name
        }
    elif isinstance(data, (int, float, str, bool)):
        return data
    elif hasattr(data, 'tolist'):  # For any array-like object with tolist method
        return data.tolist()
    elif hasattr(data, '__dict__'):
        # For custom objects, convert to string representation
        return {
            'type': 'object',
            'class': data.__class__.__name__,
            'str_repr': str(data)
        }
    else:
        # Fallback: convert to string
        return str(data)


def validate_json_serializable(data, name="data"):
    """
    Validate that data can be serialized to JSON.
    
    Args:
        data: Data to validate
        name: Name of the data for error messages
    
    Raises:
        ExperimentLoggerError: If data is not serializable
    """
    try:
        # Try to serialize the data
        json.dumps(serialize_data(data), cls=NumpyEncoder)
        return True
    except (TypeError, ValueError) as e:
        raise ExperimentLoggerError(f"Data '{name}' is not JSON serializable: {str(e)}")


class ExperimentLoggerError(Exception):
    """Custom exception for ExperimentLogger errors."""
    pass


class ExperimentLogger:
    """
    Simple class to log experiments in a JSON file using string configurations.
    """

    def __init__(self, dataset:Dict [str, Any] = None,
                 grsf_model:Dict[str, Any] = None,
                 surrogate_model:Dict[str, Any] = None,
                 local_generator:Optional[Dict[str, Any]] = None,
                 batch_generator:Optional[Dict[str, Any]] = None) -> None:
        self._dataset = dataset
        self._grsf_model = grsf_model
        self._surrogate_model = surrogate_model
        self._local_generator = local_generator
        self._batch_generator = batch_generator

    def set_dataset(self, dataset: Dict[str, Any]) -> None:
        self._dataset = dataset
        self._grsf_model = None
        self._surrogate_model = None
    
    def set_grsf_model(self, grsf_model: Dict[str, Any]) -> None:
        if self._dataset is None:
            raise ExperimentLoggerError("Dataset must be set before setting the GRSF model.")
        self._grsf_model = grsf_model
        self._surrogate_model = None

    def set_surrogate_model(self, surrogate_model: Dict[str, Any]) -> None:
        if self._grsf_model is None:
            raise ExperimentLoggerError("GRSF model must be set before setting the surrogate model.")
        self._surrogate_model = surrogate_model
    
    def set_local_generator(self, local_generator: Dict[str, Any]) -> None:
        if local_generator is None:
            self._local_generator = None
            return
        validate_json_serializable(local_generator, "local_generator")
        self._local_generator = serialize_data(local_generator)
    
    def set_batch_generator(self, batch_generator: Dict[str, Any]) -> None:
        if batch_generator is None:
            self._batch_generator = None
            return
        # Validate and serialize the batch generator data
        validate_json_serializable(batch_generator, "batch_generator")
        self._batch_generator = serialize_data(batch_generator)

    def clear_generators(self) -> None:
        """
        Clear the local and batch generators.
        """
        print("Clearing local and batch generators.")
    
    def save_experiment(self, experiment_name: str) -> None:
        """
        Save the current experiment state to a JSON file.
        """
        
        if not os.path.exists(EXPERIMENTS_DIR):
            os.makedirs(EXPERIMENTS_DIR)
        
        if not experiment_name:
            experiment_name = "experiment"
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = os.path.join(EXPERIMENTS_DIR, filename)
        print(f"DEBUG: Saving experiment to {filepath}")

        # Serialize all data to handle NumPy arrays and other non-JSON types
        experiment_data = {
            "dataset": serialize_data(self._dataset),
            "grsf_model": serialize_data(self._grsf_model),
            "surrogate_model": serialize_data(self._surrogate_model),
            "local_generator": serialize_data(self._local_generator),
            "batch_generator": serialize_data(self._batch_generator),
            "timestamp": time.time(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Validate data before saving
        validate_json_serializable(experiment_data, "experiment_data")

        try:
            with open(filepath, 'w') as f:
                json.dump(experiment_data, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
            
            print(f"Experiment saved to {filepath}")
            self._local_generator = None
            self._batch_generator = None
        except (TypeError, ValueError, IOError) as e:
            raise ExperimentLoggerError(f"Failed to save experiment: {str(e)}")  
    
    def load_experiment(self, filepath: str) -> None:
        """
        Load an experiment from a JSON file.
        """
        if not os.path.exists(filepath):
            raise ExperimentLoggerError(f"Experiment file {filepath} does not exist.")
        
        with open(filepath, 'r') as f:
            experiment_data = json.load(f)
        
        self._dataset = experiment_data.get("dataset")
        self._grsf_model = experiment_data.get("grsf_model")
        self._surrogate_model = experiment_data.get("surrogate_model")
        self._local_generator = experiment_data.get("local_generator")
        self._batch_generator = experiment_data.get("batch_generator")
        
        print(f"Experiment loaded from {filepath}")

    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get the current experiment information.
        """
        return {
            "dataset": self._dataset,
            "grsf_model": self._grsf_model,
            "surrogate_model": self._surrogate_model,
            "local_generator": self._local_generator,
            "batch_generator": self._batch_generator
        }
    
    def debug_serialization(self, data, name="data", max_depth=3, current_depth=0):
        """
        Debug method to identify non-serializable objects in the data structure.
        
        Args:
            data: Data to analyze
            name: Name of the current data element
            max_depth: Maximum depth to analyze
            current_depth: Current recursion depth
        """
        if current_depth > max_depth:
            return
        
        indent = "  " * current_depth
        try:
            json.dumps(serialize_data(data), cls=NumpyEncoder)
            print(f"{indent}✓ {name}: {type(data)} - Serializable")
        except (TypeError, ValueError) as e:
            print(f"{indent}✗ {name}: {type(data)} - ERROR: {str(e)}")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    self.debug_serialization(value, f"{name}[{key}]", max_depth, current_depth + 1)
            elif isinstance(data, (list, tuple)):
                for i, item in enumerate(data):
                    if i < 5:  # Only check first 5 items to avoid spam
                        self.debug_serialization(item, f"{name}[{i}]", max_depth, current_depth + 1)

class ExperimentManager:
    """
    Class for managing and displaying experiment results.
    """
    
    def __init__(self):
        if not os.path.exists(EXPERIMENTS_DIR):
            os.makedirs(EXPERIMENTS_DIR)
    
    def list_experiments(self) -> list:
        """
        List all available experiment files.
        
        Returns:
            List of dictionaries with experiment metadata
        """
        experiments = []
        if not os.path.exists(EXPERIMENTS_DIR):
            return experiments
        
        for file in os.listdir(EXPERIMENTS_DIR):
            if file.endswith('.json'):
                try:
                    filepath = os.path.join(EXPERIMENTS_DIR, file)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Extract metadata
                    experiment_info = {
                        'filename': file,
                        'name': file[:-5],  # Remove .json extension
                        'filepath': filepath,
                        'dataset_name': data.get('dataset', {}).get('dataset_name', 'Unknown'),
                        'timestamp': data.get('timestamp', 0),
                        'created_at': data.get('created_at', 'Unknown'),
                        'has_local_generator': data.get('local_generator') is not None,
                        'has_batch_generator': data.get('batch_generator') is not None,
                        'grsf_accuracy': data.get('grsf_model', {}).get('accuracy', 'N/A'),
                        'surrogate_accuracy': data.get('surrogate_model', {}).get('accuracy', 'N/A'),
                        'file_size': os.path.getsize(filepath)
                    }
                    experiments.append(experiment_info)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not read experiment file {file}: {e}")
                    continue
        
        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x['timestamp'], reverse=True)
        return experiments
    
    def load_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """
        Load an experiment by name.
        
        Args:
            experiment_name: Name of the experiment (with or without .json extension)
            
        Returns:
            Dictionary containing the experiment data
        """
        if not experiment_name.endswith('.json'):
            experiment_name += '.json'
        
        filepath = os.path.join(EXPERIMENTS_DIR, experiment_name)
        if not os.path.exists(filepath):
            raise ExperimentLoggerError(f"Experiment file {experiment_name} not found.")
        
        try:
            with open(filepath, 'r') as f:
                experiment_data = json.load(f)
            return experiment_data
        except json.JSONDecodeError as e:
            raise ExperimentLoggerError(f"Invalid JSON in experiment file {experiment_name}: {e}")
        except IOError as e:
            raise ExperimentLoggerError(f"Could not read experiment file {experiment_name}: {e}")
    
    def delete_experiment(self, experiment_name: str) -> None:
        """
        Delete an experiment file.
        
        Args:
            experiment_name: Name of the experiment (with or without .json extension)
        """
        if not experiment_name.endswith('.json'):
            experiment_name += '.json'
        
        filepath = os.path.join(EXPERIMENTS_DIR, experiment_name)
        if not os.path.exists(filepath):
            raise ExperimentLoggerError(f"Experiment file {experiment_name} not found.")
        
        try:
            os.remove(filepath)
            print(f"Experiment {experiment_name} deleted successfully.")
        except IOError as e:
            raise ExperimentLoggerError(f"Could not delete experiment file {experiment_name}: {e}")
    
    def get_experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """
        Get a summary of an experiment without loading all data.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary with experiment summary
        """
        experiment_data = self.load_experiment(experiment_name)
        
        summary = {
            'name': experiment_name.replace('.json', ''),
            'dataset': experiment_data.get('dataset', {}),
            'grsf_model': {
                'accuracy': experiment_data.get('grsf_model', {}).get('accuracy', 'N/A'),
                'parameters': experiment_data.get('grsf_model', {}).get('parameters', {})
            },
            'surrogate_model': {
                'accuracy': experiment_data.get('surrogate_model', {}).get('accuracy', 'N/A'),
                'architecture': experiment_data.get('surrogate_model', {}).get('model_architecture', 'Unknown'),
                'parameters': experiment_data.get('surrogate_model', {}).get('parameters', {})
            },
            'generation_type': 'local' if experiment_data.get('local_generator') else 'batch',
            'timestamp': experiment_data.get('timestamp', 0),
            'created_at': experiment_data.get('created_at', 'Unknown')
        }
        
        # Add generator-specific info
        if experiment_data.get('local_generator'):
            local_gen = experiment_data['local_generator']
            summary['local_generation'] = {
                'generation_time': local_gen.get('generation_time', 'N/A'),
                'base_class': local_gen.get('base_class', 'N/A'),
                'target_class': local_gen.get('target_class', 'N/A'),
                'parameters': local_gen.get('parameters', {}),
                'is_generated': local_gen.get('is_generated', False)
            }
        
        if experiment_data.get('batch_generator'):
            batch_gen = experiment_data['batch_generator']
            summary['batch_generation'] = {
                'stats': batch_gen.get('stats', {}),
                'parameters': batch_gen.get('parameters', {}),
                'num_generated': len(batch_gen.get('counterfactuals', []))
            }
        
        return summary
    
    def export_experiment_summary(self, experiment_name: str) -> str:
        """
        Export experiment summary as formatted text.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Formatted string summary
        """
        summary = self.get_experiment_summary(experiment_name)
        
        text = f"# Experiment: {summary['name']}\n\n"
        text += f"**Created:** {summary['created_at']}\n\n"
        
        # Dataset info
        dataset = summary['dataset']
        text += f"## Dataset\n"
        text += f"- **Name:** {dataset.get('dataset_name', 'Unknown')}\n"
        text += f"- **Samples:** {dataset.get('num_samples', 'N/A')}\n"
        text += f"- **Features:** {dataset.get('sample_size', 'N/A')}\n"
        text += f"- **Classes:** {dataset.get('num_classes', 'N/A')}\n\n"
        
        # GRSF model info
        grsf = summary['grsf_model']
        text += f"## GRSF Model\n"
        text += f"- **Accuracy:** {grsf['accuracy']}\n"
        if grsf['parameters']:
            text += f"- **Parameters:**\n"
            for key, value in grsf['parameters'].items():
                text += f"  - {key}: {value}\n"
        text += "\n"
        
        # Surrogate model info
        surrogate = summary['surrogate_model']
        text += f"## Surrogate Model\n"
        text += f"- **Architecture:** {surrogate['architecture']}\n"
        text += f"- **Accuracy:** {surrogate['accuracy']}\n"
        if surrogate['parameters']:
            text += f"- **Parameters:**\n"
            for key, value in surrogate['parameters'].items():
                text += f"  - {key}: {value}\n"
        text += "\n"
        
        # Generation info
        text += f"## Counterfactual Generation\n"
        text += f"- **Type:** {summary['generation_type']}\n"
        
        if 'local_generation' in summary:
            local = summary['local_generation']
            text += f"- **Generation Time:** {local['generation_time']} seconds\n"
            text += f"- **Base Class:** {local['base_class']}\n"
            text += f"- **Target Class:** {local['target_class']}\n"
            text += f"- **Successfully Generated:** {local['is_generated']}\n"
        
        if 'batch_generation' in summary:
            batch = summary['batch_generation']
            text += f"- **Number Generated:** {batch['num_generated']}\n"
            if batch['stats']:
                text += f"- **Statistics:**\n"
                for key, value in batch['stats'].items():
                    text += f"  - {key}: {value}\n"
        
        return text
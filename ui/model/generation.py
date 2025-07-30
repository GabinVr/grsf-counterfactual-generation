import os
import sys
import torch
import numpy as np
import streamlit as st
import time
from typing import Dict, Any, Tuple, Optional, List
UI_ROOT = os.path.dirname(os.path.dirname(__file__))
if UI_ROOT not in sys.path:
    sys.path.append(UI_ROOT)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from gen import CounterFactualCrafting
from counterfactual import counterfactual_local_generation, counterfactual_batch_generation, get_target_from_base_class
import wildboar.distance as wb_distance

class LocalCounterfactualGeneratorObject:
    def __init__(self):
        """Initialize the local counterfactual generator object"""
        self._grsf_model = None
        self._surrogate_model = None
        self._base_sample = None
        self._base_class = None
        self._target_sample = None
        self._target_class = None
        self._binary_mask = None
        self._parameters = None
        self._counterfactual = None
        self._training_progress = ""
        self._is_generated = False
        self._generation_time = None
    
    def is_empty(self) -> bool:
        """Check if the counterfactual has been generated"""
        return self._counterfactual is None
    
    def has_changed(self, parameters: dict) -> bool:
        """Check if the generation parameters have changed"""
        if self._parameters is None:
            return True

        return self._parameters != parameters
    
    def clear(self) -> None:
        """Reset the counterfactual generator"""
        self._counterfactual = None
        self._training_progress = ""
        self._is_generated = False
        self._generation_time = None
    
    def get_counterfactual(self) -> Optional[torch.Tensor]:
        """Get the generated counterfactual"""
        if self._counterfactual is None:
            raise ValueError("Counterfactual has not been generated yet.")
        return self._counterfactual
    
    def get_counterfactual_triplet(self) -> Optional[Tuple]:
        """Get the counterfactual triplet (counterfactual, target, base)"""
        if self._counterfactual is None:
            return None
        
        base_np = self._base_sample.detach().numpy() if torch.is_tensor(self._base_sample) else self._base_sample
        target_np = self._target_sample.detach().numpy() if torch.is_tensor(self._target_sample) else self._target_sample
        cf_np = self._counterfactual.detach().numpy() if torch.is_tensor(self._counterfactual) else self._counterfactual
        
        return (cf_np, (target_np, self._target_class), (base_np, self._base_class))
    
    def get_training_progress(self) -> str:
        """Get the training progress log"""
        return self._training_progress
    
    def set_models(self, grsf_model, surrogate_model) -> None:
        """Set the GRSF and surrogate models used for generation"""
        if grsf_model is None:
            raise ValueError("GRSF model cannot be None.")
        if surrogate_model is None:
            raise ValueError("Surrogate model cannot be None.")
            
        self._grsf_model = grsf_model
        self._surrogate_model = surrogate_model
        self.clear()  # Clear previous counterfactuals when models change
    
    def set_base_sample(self, sample, class_label: int) -> None:
        """Set the base sample from which to generate a counterfactual"""
        if sample is None:
            raise ValueError("Base sample cannot be None.")
            
        # Convert to tensor if not already
        if not torch.is_tensor(sample):
            sample = torch.tensor(sample, dtype=torch.float32)
            
        self._base_sample = sample
        self._base_class = class_label
        self.clear()  # Clear previous counterfactuals when base sample changes
    
    def set_target_sample(self, sample, class_label: int) -> None:
        """Set the target sample/class for counterfactual generation"""
        if sample is None:
            raise ValueError("Target sample cannot be None.")
            
        # Convert to tensor if not already
        if not torch.is_tensor(sample):
            sample = torch.tensor(sample, dtype=torch.float32)
            
        self._target_sample = sample
        self._target_class = class_label
        self.clear()  # Clear previous counterfactuals when target sample changes
    
    def set_binary_mask(self, mask) -> None:
        """Set the binary mask defining which parts of the time series can be modified"""
        if mask is None:
            self._binary_mask = None
            return
            
        # Convert to tensor if not already
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.float32)
            
        self._binary_mask = mask
        self.clear()  # Clear previous counterfactuals when mask changes
    
    def set_parameters(self, parameters: dict) -> None:
        """Set the parameters for counterfactual generation"""
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary.")
            
        required_keys = ["epochs", "learning_rate", "beta"]
        for key in required_keys:
            if key not in parameters:
                raise ValueError(f"Missing required parameter: {key}")
                
        self._parameters = parameters.copy()
        self.clear()  # Clear previous counterfactuals when parameters change
    
    def select_random_target(self, split_dataset: Tuple) -> None:
        """Select a random target sample from a different class than the base sample"""
        if self._base_sample is None or self._base_class is None:
            raise ValueError("Base sample must be set before selecting a random target.")
        
        _, _, X_test, y_test = split_dataset
        target_sample, target_class = get_target_from_base_class(
            self._base_class, y_test, X_test
        )
        
        self.set_target_sample(target_sample, target_class)
    
    def _training_callback(self, epoch: int, loss: float) -> str:
        """Callback for tracking training progress"""
        progress = f"Epoch {epoch + 1}: Loss = {loss:.4f}\n"
        self._training_progress+= progress
        return progress
    
    def get_base_sample(self):
        """Get the base sample used for counterfactual generation"""
        if self._base_sample is None:
            raise ValueError("Base sample has not been set.")
        return self._base_sample
    
    def get_target_sample(self):
        """Get the target sample used for counterfactual generation"""
        if self._target_sample is None:
            raise ValueError("Target sample has not been set.")
        return self._target_sample

    def get_base_class(self) -> int:
        """Get the class label of the base sample"""
        if self._base_class is None:
            raise ValueError("Base class has not been set.")
        return self._base_class
    
    def get_target_class(self) -> int:
        """Get the class label of the target sample"""
        if self._target_class is None:
            raise ValueError("Target class has not been set.")
        return self._target_class
    
    def generate(self) -> bool:
        """Generate the local counterfactual"""
        # Validate input
        if self._grsf_model is None or self._surrogate_model is None:
            raise ValueError("GRSF and surrogate models must be set before generating counterfactuals.")
            
        if self._base_sample is None or self._target_sample is None:
            raise ValueError("Base and target samples must be set before generating counterfactuals.")
            
        if self._parameters is None:
            raise ValueError("Parameters must be set before generating counterfactuals.")
            
        # Reset training progress
        self._training_progress = ""
        print(f"DEBUG: {str(self)}")
        print(f"DEBUG: {type(self._surrogate_model)}")
        crafter = CounterFactualCrafting(
            self._grsf_model,
            self._surrogate_model
        )

        start_time = time.time()
        print(f"DEBUG: Starting counterfactual generation with parameters: {self._parameters}")
        # Generate the counterfactual
        self._counterfactual = crafter.generate_local_counterfactuals(
            self._target_sample,
            self._base_sample,
            self._base_class,
            self._binary_mask,
            lr =self._parameters["learning_rate"],
            epochs=self._parameters["epochs"],
            beta=self._parameters["beta"],
            debug = True,
            training_callback=self._training_callback,
        )
        self._counterfactual = self._counterfactual.detach().cpu().numpy() if torch.is_tensor(self._counterfactual) else self._counterfactual
        print(f"DEBUG: Counterfactual generation completed in {time.time() - start_time:.2f} seconds")
        # End timing
        self._generation_time = time.time() - start_time
        
        # Check if generation was successful
        if self._counterfactual is None:
            return False
            
        self._is_generated = True
        return True
            
    
    def is_valid(self) -> bool:
        """Check if the generated counterfactual is valid (changes the GRSF model's prediction)"""
        if not self._is_generated or self._counterfactual is None:
            return False
            
        try:
            base_pred = self._grsf_model.predict(
                self._base_sample.reshape(1, -1)
            )[0]
            
            cf_pred = self._grsf_model.predict(
                self._counterfactual.reshape(1, -1)
            )[0]
            
            return base_pred != cf_pred
            
        except Exception as e:
            self._training_progress += f"Validity check error: {str(e)}\n"
            return False
    
    def has_counterfactual(self) -> bool:
        """Check if a counterfactual has been generated"""
        return self._is_generated and self._counterfactual is not None
    
    def has_target_sample(self) -> bool:
        """Check if a target sample has been set"""
        return self._target_sample is not None and self._target_class is not None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the counterfactual"""
        if not self._is_generated or self._counterfactual is None:
            return {"generated": False}
            
        # Calculate various metrics
        import numpy as np
        import wildboar.distance as wb_distance
        
        cf = self._counterfactual.detach().numpy() if torch.is_tensor(self._counterfactual) else self._counterfactual
        base = self._base_sample.detach().numpy() if torch.is_tensor(self._base_sample) else self._base_sample
        target = self._target_sample.detach().numpy() if torch.is_tensor(self._target_sample) else self._target_sample
        
        stats = {
            "generated": True,
            "base_euclidean": float(np.linalg.norm(cf - base)),
            "target_euclidean": float(np.linalg.norm(cf - target)),
            "base_dtw": float(wb_distance.dtw.dtw_distance(cf, base, r=1.0)),
            "target_dtw": float(wb_distance.dtw.dtw_distance(cf, target, r=1.0)),
            "sparsity": float(np.sum(cf != base) / cf.size * 100),
            "valid": self.is_valid(),
            "generation_time": self._generation_time
        }
        
        return stats
    
    def get_info(self) -> Dict[str, Any]:
        # convert data to numpy if it's a tensor
        base_np = self._base_sample.detach().numpy() if torch.is_tensor(self._base_sample) else self._base_sample
        target_np = self._target_sample.detach().numpy() if torch.is_tensor(self._target_sample) else self._target_sample
        cf_np = self._counterfactual.detach().numpy() if torch.is_tensor(self._counterfactual) else self._counterfactual
        # return a dictionary with all relevant information
        return {
            "base_sample": base_np,
            "base_class": self._base_class,
            "target_sample": target_np,
            "target_class": self._target_class,
            "counterfactual": cf_np,
            "binary_mask": self._binary_mask,
            "parameters": self._parameters,
            "generation_time": self._generation_time,
            "is_generated": self._is_generated,
            "training_progress": self._training_progress,
            "stats": self.get_stats()
        }

    def __str__(self) -> str:
        """String representation of the counterfactual generator"""
        if not self._is_generated:
            return f"""Local counterfactual: Not generated yet, 
                    - Base class: {self._base_class}
                    - Target class: {self._target_class}
                    - Binary mask: {self._binary_mask}
                    - Parameters: {self._parameters}"""
        
        stats = self.get_stats()
        
        validity_str = "✅ Valid" if stats["valid"] else "❌ Invalid"
        
        return (
            f"Local counterfactual:\n"
            f"- {validity_str}\n"
            f"- Base class: {self._base_class}\n"
            f"- Target class: {self._target_class}\n"
            f"- Euclidean distance to base: {stats['base_euclidean']:.2f}\n"
            f"- Euclidean distance to target: {stats['target_euclidean']:.2f}\n"
            f"- Generation time: {self._generation_time:.2f} seconds\n"
            f"- Parameters: {self._parameters}\n"
        )
    

class GlobalCounterfactualGeneratorObject:
    def __init__(self):
        """Initialize the global counterfactual generator object"""
        self._grsf_model = None
        self._surrogate_model = None
        self._split_dataset = None
        self._parameters = None
        self._counterfactuals = None
        self._training_progress = []
        self._is_generated = False
        self._generation_time = None
        self._nb_samples = 10  # Default number of samples
    
    def is_empty(self) -> bool:
        """Check if counterfactuals have been generated"""
        return self._counterfactuals is None
    
    def has_changed(self, parameters: dict) -> bool:
        """Check if the generation parameters have changed"""
        if self._parameters is None:
            return True
        return self._parameters != parameters
    
    def clear(self) -> None:
        """Reset the counterfactual generator"""
        self._counterfactuals = None
        self._training_progress = []
        self._is_generated = False
        self._generation_time = None
    
    def get_counterfactuals(self) -> Optional[List]:
        """Get the generated counterfactuals list"""
        if self._counterfactuals is None:
            raise ValueError("Counterfactuals have not been generated yet.")
        return self._counterfactuals
    
    def get_training_progress(self) -> str:
        """Get the training progress log"""
        return self._training_progress
    
    def set_models(self, grsf_model, surrogate_model) -> None:
        """Set the GRSF and surrogate models used for generation"""
        if grsf_model is None:
            raise ValueError("GRSF model cannot be None.")
        if surrogate_model is None:
            raise ValueError("Surrogate model cannot be None.")
            
        self._grsf_model = grsf_model
        self._surrogate_model = surrogate_model
        self.clear()  # Clear previous counterfactuals when models change
    
    def set_split_dataset(self, split_dataset: Tuple) -> None:
        """Set the split dataset for counterfactual generation"""
        if split_dataset is None:
            raise ValueError("Split dataset cannot be None.")
            
        self._split_dataset = split_dataset
        self.clear()  # Clear previous counterfactuals when dataset changes
    
    def set_parameters(self, parameters: dict, nb_samples: int = 10) -> None:
        """Set the parameters for counterfactual generation"""
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary.")
            
        required_keys = ["epochs", "learning_rate", "beta"]
        for key in required_keys:
            if key not in parameters:
                raise ValueError(f"Missing required parameter: {key}")
                
        self._parameters = parameters.copy()
        self._nb_samples = nb_samples
        self.clear()  # Clear previous counterfactuals when parameters change
    
    def _training_callback(self, data: str) -> str:
        print(f"DEBUG: Training callback data: {data}")
        self._training_progress.append(data)
        return data
    
    def generate(self) -> bool:
        """Generate batch counterfactuals"""
        # Validate input
        if self._grsf_model is None or self._surrogate_model is None:
            raise ValueError("GRSF and surrogate models must be set before generating counterfactuals.")
            
        if self._split_dataset is None:
            raise ValueError("Split dataset must be set before generating counterfactuals.")
            
        if self._parameters is None:
            raise ValueError("Parameters must be set before generating counterfactuals.")
        
        # Reset training progress
        self._training_progress = []
        
        start_time = time.time()
        print(f"DEBUG: Starting batch counterfactual generation with {self._nb_samples} samples")
        print(f"DEBUG: Parameters: {self._parameters}")
        
        try:
            # Generate batch counterfactuals
            self._counterfactuals = counterfactual_batch_generation(
                grsf_classifier=self._grsf_model,
                nn_classifier=self._surrogate_model,
                split_dataset=self._split_dataset,
                nb_samples=self._nb_samples,
                epochs=self._parameters["epochs"],
                lr=self._parameters["learning_rate"],
                beta=self._parameters["beta"],
                debug=True,
                training_callback=self._training_callback
            )
            
            # End timing
            self._generation_time = time.time() - start_time
            
            # Check if generation was successful
            if self._counterfactuals is None or len(self._counterfactuals) == 0:
                print(f"DEBUG: Batch generation failed - no counterfactuals generated")
                return False
                
            self._is_generated = True
            print(f"DEBUG: Batch counterfactual generation completed in {self._generation_time:.2f} seconds")
            print(f"DEBUG: Generated {len(self._counterfactuals)} counterfactuals")
            return True
            
        except Exception as e:
            print(f"ERROR: Batch counterfactual generation failed: {str(e)}")
            self._training_progress.append(f"Batch generation error: {str(e)}")
            return False
            
    def is_valid(self) -> bool:
        """Check if the generated counterfactuals are valid"""
        if not self._is_generated or self._counterfactuals is None:
            return False
            
        try:
            valid_count = 0
            total_count = len(self._counterfactuals)
            
            for counterfactual, target, base in self._counterfactuals:
                base_sample = base[0]
                cf_sample = counterfactual.detach().numpy() if torch.is_tensor(counterfactual) else counterfactual
                
                base_pred = self._grsf_model.predict(base_sample.reshape(1, -1))[0]
                cf_pred = self._grsf_model.predict(cf_sample.reshape(1, -1))[0]
                
                if base_pred != cf_pred:
                    valid_count += 1
            
            return valid_count > 0  # At least one valid counterfactual
            
        except Exception as e:
            self._training_progress.append(f"Validity check error: {str(e)}")
            return False
    
    def has_counterfactuals(self) -> bool:
        """Check if counterfactuals have been generated"""
        return self._is_generated and self._counterfactuals is not None and len(self._counterfactuals) > 0
            
    def get_stats(self, mode: str = "basic") -> Dict[str, Any]:
        """
        Get statistics about the generated counterfactuals.
        
        Args:
            mode: Statistics mode - "basic" for summary statistics, "all" for per-sample details
            
        Returns:
            Dictionary containing statistics based on the specified mode
        """
        if not self._is_generated or self._counterfactuals is None:
            return {"generated": False}
        
        if len(self._counterfactuals) == 0:
            return {"generated": True, "total_count": 0}
            
        # Calculate metrics for all counterfactuals
        metrics_data = self._calculate_all_metrics()
        
        if mode == "all":
            return self._format_detailed_stats(metrics_data)
        
        return self._format_summary_stats(metrics_data)
    
    def _calculate_all_metrics(self) -> Dict[str, List]:
        """Calculate all metrics for counterfactuals and return organized data."""
        metrics = {
            'base_euclidean_distances': [],
            'target_euclidean_distances': [],
            'base_dtw_distances': [],
            'target_dtw_distances': [],
            'sparsities': [],
            'validities': []
        }
        
        for counterfactual, target, base in self._counterfactuals:
            # Convert tensors to numpy arrays consistently
            cf_np = self._tensor_to_numpy(counterfactual)
            base_np = self._tensor_to_numpy(base[0])
            target_np = self._tensor_to_numpy(target[0])
            
            # Calculate Euclidean distances
            metrics['base_euclidean_distances'].append(
                float(np.linalg.norm(cf_np - base_np))
            )
            metrics['target_euclidean_distances'].append(
                float(np.linalg.norm(cf_np - target_np))
            )
            
            # Calculate DTW distances
            try:
                base_dtw = float(wb_distance.dtw.dtw_distance(cf_np, base_np, r=1.0))
                target_dtw = float(wb_distance.dtw.dtw_distance(cf_np, target_np, r=1.0))
            except Exception as e:
                # Fallback to Euclidean if DTW fails
                base_dtw = metrics['base_euclidean_distances'][-1]
                target_dtw = metrics['target_euclidean_distances'][-1]
                
            metrics['base_dtw_distances'].append(base_dtw)
            metrics['target_dtw_distances'].append(target_dtw)
            
            # Calculate sparsity (percentage of changed elements)
            total_elements = cf_np.size
            changed_elements = np.sum(cf_np != base_np)
            sparsity = float(changed_elements / total_elements * 100) if total_elements > 0 else 0.0
            metrics['sparsities'].append(sparsity)
            
            # Check validity (class change)
            is_valid = self._check_counterfactual_validity(cf_np, base_np)
            metrics['validities'].append(is_valid)
        
        return metrics
    
    def _tensor_to_numpy(self, tensor_data) -> np.ndarray:
        """Safely convert tensor to numpy array."""
        if torch.is_tensor(tensor_data):
            return tensor_data.detach().numpy()
        return np.asarray(tensor_data)
    
    def _check_counterfactual_validity(self, cf_np: np.ndarray, base_np: np.ndarray) -> bool:
        """Check if counterfactual is valid (causes class change)."""
        if self._grsf_model is None:
            return False
            
        try:
            # Reshape for prediction if needed
            cf_reshaped = cf_np.reshape(1, -1)
            base_reshaped = base_np.reshape(1, -1)
            
            base_pred = self._grsf_model.predict(base_reshaped)[0]
            cf_pred = self._grsf_model.predict(cf_reshaped)[0]
            
            return base_pred != cf_pred
            
        except (IndexError, AttributeError, ValueError) as e:
            # Log the error if needed for debugging
            return False
    
    def _format_detailed_stats(self, metrics_data: Dict[str, List]) -> Dict[str, Any]:
        """Format statistics for 'all' mode with per-sample details."""
        stats = {}
        num_samples = len(self._counterfactuals)
        
        for i in range(num_samples):
            stats[i] = {
                "generated": True,
                "base_euclidean": metrics_data['base_euclidean_distances'][i],
                "target_euclidean": metrics_data['target_euclidean_distances'][i],
                "base_dtw": metrics_data['base_dtw_distances'][i],
                "target_dtw": metrics_data['target_dtw_distances'][i],
                "sparsity": metrics_data['sparsities'][i],
                "valid": metrics_data['validities'][i],
                "generation_time": self._generation_time,
            }
        
        return stats
    
    def _format_summary_stats(self, metrics_data: Dict[str, List]) -> Dict[str, Any]:
        """Format summary statistics for 'basic' mode."""
        total_count = len(self._counterfactuals)
        valid_count = sum(metrics_data['validities'])
        
        # Calculate averages safely
        avg_base_euclidean = float(np.mean(metrics_data['base_euclidean_distances']))
        avg_target_euclidean = float(np.mean(metrics_data['target_euclidean_distances']))
        avg_sparsity = float(np.mean(metrics_data['sparsities']))
        
        # Calculate validity rate safely
        validity_rate = float(valid_count / total_count * 100) if total_count > 0 else 0.0
        
        return {
            "generated": True,
            "total_count": total_count,
            "valid_count": valid_count,
            "validity_rate": validity_rate,
            "avg_base_euclidean": avg_base_euclidean,
            "avg_target_euclidean": avg_target_euclidean,
            "avg_sparsity": avg_sparsity,
            "generation_time": self._generation_time,
            "parameters": self._parameters
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about generated counterfactuals."""
        if not self._is_generated or self._counterfactuals is None:
            return {
                "counterfactuals": [],
                "parameters": self._parameters,
                "generation_time": self._generation_time,
                "is_generated": False,
                "training_progress": self._training_progress,
                "stats": {}
            }
        
        # Convert counterfactuals to numpy format
        cf_list = []
        for counterfactual, target, base in self._counterfactuals:
            cf_np = self._tensor_to_numpy(counterfactual)
            target_np = self._tensor_to_numpy(target[0])
            base_np = self._tensor_to_numpy(base[0])
            cf_list.append((cf_np, (target_np, target[1]), (base_np, base[1])))
        
        return {
            "counterfactuals": cf_list,
            "parameters": self._parameters,
            "generation_time": self._generation_time,
            "is_generated": self._is_generated,
            "training_progress": self._training_progress,
            "stats": self.get_stats(mode="all")
        }
    
    def __str__(self) -> str:
        """String representation of the batch counterfactual generator"""
        if not self._is_generated:
            return f"""Batch counterfactual generator: Not generated yet
                    - Number of samples: {self._nb_samples}
                    - Parameters: {self._parameters}"""
        
        stats = self.get_stats()
        
        return (
            f"Batch counterfactual generator:\n"
            f"- Generated: {stats['total_count']} counterfactuals\n"
            f"- Valid: {stats['valid_count']} ({stats['validity_rate']:.1f}%)\n"
            f"- Avg euclidean distance to base: {stats['avg_base_euclidean']:.2f}\n"
            f"- Avg euclidean distance to target: {stats['avg_target_euclidean']:.2f}\n"
            f"- Avg sparsity: {stats['avg_sparsity']:.2f}%\n"
            f"- Generation time: {stats['generation_time']:.2f} seconds\n"
            f"- Parameters: {stats['parameters']}\n"
        )

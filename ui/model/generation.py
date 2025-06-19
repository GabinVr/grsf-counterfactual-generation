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
from counterfactual import counterfactual_local_generation, get_target_from_base_class


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
        self._training_progress += progress
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
    pass
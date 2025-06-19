import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from typing import Dict, Any, Optional, Tuple, List, Union
import hashlib
import random

class SampleSelectComponent:
    """
    Component for selecting part of a time series sample on a plot.
    """

    def __init__(self, sample: Union[pd.Series, np.ndarray, torch.Tensor], 
                 target: Union[pd.Series, np.ndarray, torch.Tensor], 
                 sample_class: str, target_class: str):

        self.sample = self._convert_to_numpy(sample)
        self.target = self._convert_to_numpy(target)
        self.sample_length = len(self.sample)
        self.sample_class = sample_class
        self.target_class = target_class
        self.counterfactual = None
        self.selected_points = None
        self.binary_mask = None
        self._show_counterfactual = False
        

        sample_hash = str(hash(self.sample.tobytes() if len(self.sample) > 0 else "empty"))
        target_hash = str(hash(self.target.tobytes() if len(self.target) > 0 else "empty"))
        self._component_id = hashlib.md5(f"{sample_hash}_{target_hash}_{sample_class}_{target_class}".encode()).hexdigest()[:8]
    
    def _convert_to_numpy(self, data):
        """Convert different data types to numpy array for consistent handling."""
        if isinstance(data, pd.Series):
            return data.to_numpy()
        elif torch.is_tensor(data):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def set_counterfactual(self, counterfactual: Union[pd.Series, np.ndarray, torch.Tensor]) -> None:
        """
        Set a counterfactual to be displayed on the plot.
        
        :param counterfactual: The counterfactual data
        """
        if counterfactual is not None:
            self.counterfactual = self._convert_to_numpy(counterfactual)
            self._show_counterfactual = True
        else:
            self.counterfactual = None
            self._show_counterfactual = False
    
    def clear_counterfactual(self) -> None:
        self.counterfactual = None
        self._show_counterfactual = False
    
    def get_binary_mask(self) -> Optional[np.ndarray]:
        """Get the binary mask representing the selected region."""
        return self.binary_mask
    
    def render(self) -> None:

        if self._show_counterfactual and self.counterfactual is not None:
            combined_df = pd.DataFrame({
                "time": list(range(self.sample_length)) + list(range(self.sample_length)) + list(range(self.sample_length)),
                "value": list(self.sample) + list(self.target) + list(self.counterfactual),
                "series_type": ["Sample"] * self.sample_length + ["Target"] * self.sample_length + ["Counterfactual"] * self.sample_length,
                "class": [self.sample_class] * self.sample_length + [self.target_class] * self.sample_length + [self.target_class] * self.sample_length,
            })
        else:
            combined_df = pd.DataFrame({
                "time": list(range(self.sample_length)) + list(range(self.sample_length)),
                "value": list(self.sample) + list(self.target),
                "series_type": ["Sample"] * self.sample_length + ["Target"] * self.sample_length,
                "class": [self.sample_class] * self.sample_length + [self.target_class] * self.sample_length,
            })
        fig = px.scatter(combined_df, x="time", y="value", 
                         color="series_type",
                         title=f"Time Series Sample",
                         labels={"time": "Time", "value": "Value", "class": "Class"},
                         template="plotly_white")
        fig.update_layout(selectdirection='h',
                            dragmode='select',
        )

        fig.update_traces(mode='lines+markers',
                            marker=dict(size=5, opacity=0.7),
                            line=dict(width=1.5))

        event = st.plotly_chart(fig, 
                                key="time serie", 
                                on_select="rerun")
        
        idx_selected = event['selection']['point_indices']
        print(f"DEBUG: {idx_selected}, event={event}")
        # Create binary mask only for the original sample length
        binary_mask = np.zeros(self.sample_length)
        if idx_selected is not None and len(idx_selected) > 0:
            for idx in idx_selected:
                # Only consider indices that correspond to the original sample (not target)
                if 0 <= idx < self.sample_length:
                    binary_mask[idx] = 1
        else:
            return
        self.binary_mask = binary_mask.astype(int)
        self.selected_points = idx_selected
    
    def __str__(self):
        return f"SampleSelectComponent(id={self._component_id}, sample_length={self.sample_length}, class_label={self.sample_class}, target_class={self.target_class})"
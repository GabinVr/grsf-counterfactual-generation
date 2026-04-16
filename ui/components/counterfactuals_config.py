import streamlit as st
import sys
import os
from typing import Dict, Any
from core.AppConfig import AppConfig
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from code_editor import code_editor
import numpy as np
import torch
import wildboar.distance as wb_distance
from pyts.approximation import PiecewiseAggregateApproximation as PAA

import logging
from logging import getLogger

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import gen
import models
from counterfactual import counterfactual_batch_generation, counterfactual_local_generation, get_target_from_base_class

class CounterfactualsConfig:
    def __init__(self, grsf_model, surrogate_model, split_dataset):
        """
        Initialize the configuration for counterfactual generation.

        :param grsf_model: The GRSF model to use
        :param surrogate_model: The surrogate model for counterfactual generation
        :param split_dataset: the dataset (X_train, y_train, X_test, y_test) to use for counterfactual generation
        """
        self.grsf_model = grsf_model
        self.surrogate_model = surrogate_model
        self.split_dataset = split_dataset

        # Initialize session state keys if they don't exist
        if 'generation_params' not in st.session_state:
            st.session_state.generation_params = None
        if 'counterfactuals_list' not in st.session_state:
            st.session_state.counterfactuals_list = None
    
    def render(self):
        """
        Render the configuration UI for counterfactual generation.
        """
        tab1, tab2 = st.tabs(["🌍 Global", "📍 Local"])
        with tab1:
            self._render_global_counterfactuals()
        with tab2:
            self._render_local_counterfactuals()
    
    def _render_cf_parameters(self, prefix="global"):
        """
        Render the parameters for counterfactual generation.
        
        :param prefix: Prefix for unique keys (e.g., "global" or "local")
        """
        
        # Learning rate
        cf_lr = st.number_input(
            "Learning rate for counterfactual generation",
            min_value=0.00001,
            max_value=1.0,
            value=0.001,
            step=0.0001,
            key=f"{prefix}_cf_lr",
            help="Learning rate for the optimization process"
        )
        
        # Number of epochs
        cf_epochs = st.number_input(
            "Number of epochs for counterfactual generation",
            min_value=1,
            max_value=5000,
            value=100,
            step=1,
            key=f"{prefix}_cf_epochs",
            help="Number of epochs for the optimization process"
        )
        
        # Beta parameter
        cf_beta = st.number_input(
            "Beta parameter for counterfactual generation",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key=f"{prefix}_cf_beta",
            help="Beta parameter for the optimization process"
        )
        
        return {
            "learning_rate": cf_lr,
            "epochs": cf_epochs,
            "beta": cf_beta
        }
    
    def _render_global_counterfactuals(self):
        st.markdown("## 🔍 Counterfactuals generation parameters")
        st.info("Sample are chosen randomly from the test set. ")
        nb_samples = st.number_input("Number of counterfactuals to generate", 
                                    min_value=1, 
                                    max_value=1000, 
                                    value=10, 
                                    step=1,
                                    key="global_nb_samples")
        st.session_state.generation_params = {
            "nb_samples": nb_samples,
        }

        cf_params = self._render_cf_parameters("global")
        
        st.divider()
        if st.button("Generate Counterfactuals", key="global_generate_btn"):
            try:
                st.session_state.training_progress = ""
                st.session_state.counterfactuals_list = counterfactual_batch_generation(
                    grsf_classifier=self.grsf_model,
                    nn_classifier=self.surrogate_model,
                    split_dataset=self.split_dataset,
                    nb_samples=st.session_state.generation_params["nb_samples"],
                    epochs=cf_params['epochs'],
                    lr=cf_params['learning_rate'],
                    beta=cf_params['beta'],
                    training_callback=self._training_callback
                )
                self._render_training_trace()
                st.success(f"Counterfactuals generated successfully! Total: {len(st.session_state.counterfactuals_list)}")
                return st.session_state.counterfactuals_list
            except Exception as e:
                error_msg = f"Error during counterfactual generation: {str(e)}"
                st.error(f"❌ {error_msg}")
                getLogger(__name__).error(error_msg)
                st.session_state.counterfactuals_list = None
                return None
    
    def _render_training_trace(self):
        """
        Render the training trace of the counterfactual generation.
        """
        if st.session_state.training_progress:
            code_editor(st.session_state.training_progress, height=13, focus=False)       
        return 

    @staticmethod
    def _training_callback(epoch, loss):
        """
        Callback function to render training progress.
        
        :param epoch: Current epoch number
        :param loss: Current loss value
        """
        print(f"DEBUG Epoch {epoch + 1}: Loss = {loss:.4f}")
        st.session_state.training_progress += f"Epoch {epoch + 1}: Loss = {loss:.4f}\n"
        return
            
    def _render_local_counterfactuals(self):
        """
        Render the UI for local counterfactuals generation.
        """
        st.markdown("## 📍 Local Counterfactuals Generation")
        st.info("Local counterfactuals generation is sample-specific only. (for now)")
        # Params for local counterfactual generation
        cf_params = self._render_cf_parameters("local")
        cf_lr = cf_params['learning_rate']
        cf_epochs = cf_params['epochs']
        cf_beta = cf_params['beta']
        
        st.markdown("### Select a sample to generate counterfactuals for")
        self._render_sample_selection()
        if 'selected_sample' in st.session_state and st.session_state.selected_sample is not None:
            sample = st.session_state.selected_sample['sample']
            class_label = st.session_state.selected_sample['class_label']
            binary_mask = st.session_state.selected_sample['binary_mask']
            
            if st.button("Generate Local Counterfactuals", key="local_generate_btn"):
                local_counterfactual = self._generate_local_counterfactuals_from_selection(
                    sample=sample,
                    class_label=class_label,
                    binary_mask=binary_mask,
                    epochs=cf_epochs,
                    lr=cf_lr,
                    beta=cf_beta
                )
                if local_counterfactual is not None:
                    st.session_state.local_counterfactual = local_counterfactual
                    st.success("Local counterfactual generated successfully!")
                    st.session_state.counterfactuals_list = [local_counterfactual]  # Update the list with the new counterfactual
                    self._render_validity_count("local", st.session_state.counterfactuals_list)
                    st.rerun()
                    return local_counterfactual
                else:
                    st.error("❌ Failed to generate local counterfactual. Please check the selection and try again.")
            else:
                st.warning("Please select a sample and a region to generate local counterfactuals.")
    

    def _render_validity_count(self, prefix, counterfactuals):
        """
        Render the validity count for the generated counterfactual.
        
        :param prefix: Prefix for unique keys (e.g., "global" or "local")
        :param counterfactuals: The generated local counterfactual
        """
        if counterfactuals is None:
            st.error("❌ No counterfactual generated.")
            return
        
        # Check validity
        grsf_classifier = self.grsf_model
        if grsf_classifier is None:
            st.error("❌ GRSF model not available for validity check.")
            return
        
        nb_valid = 0
        for cf_triplet in counterfactuals:
            counterfactual, target, base = cf_triplet
            if not torch.is_tensor(counterfactual):
                counterfactual = torch.tensor(counterfactual, dtype=torch.float32)
            if not torch.is_tensor(target[0]):
                target = (torch.tensor(target[0], dtype=torch.float32), target[1])
            if not torch.is_tensor(base[0]):
                base = (torch.tensor(base[0], dtype=torch.float32), base[1])
            
            # Check validity
            try:
                if grsf_classifier.predict(counterfactual.reshape(1, -1)) != grsf_classifier.predict(base[0].reshape(1, -1)):
                    nb_valid += 1
            except Exception as e:
                st.error(f"❌ Error during validity check: {str(e)}")
        
        st.divider()
        st.markdown(f"### Validity Check for {prefix} Counterfactuals")
        st.markdown(f"Valid Counterfactuals: {nb_valid} out of {len(counterfactuals)}")

    def _render_sample_selection(self):
        """
        Render the sample selection UI for local counterfactuals generation.
        """
        if 'split_dataset' not in st.session_state or st.session_state.split_dataset is None:
            st.error("❌ No dataset loaded. Please load a dataset first.")
            return
        
        _, _, X_test, y_test = st.session_state.split_dataset
        
        if len(X_test) == 0:
            st.warning("⚠️ No test samples available for local counterfactuals generation.")
            return
        
        selected_idx = st.selectbox(
            "Select a base sample from the test set:",
            range(len(X_test)),
            format_func=lambda x: f" {x + 1} (Class {y_test[x]})",
            key="local_sample_selection"
        )

        # let the user choose a target sample/class
        if selected_idx is not None:
            st.markdown(f"Select a target sample:")
            rand_select = st.toggle("🌈 Randomly select a target sample", key="random_target_sample_toggle", value=True)
        
            if rand_select and "selected_target" not in st.session_state:
                _, _, X_test, y_test = st.session_state.split_dataset
                target_sample, target_class = get_target_from_base_class(y_test[selected_idx], y_test, X_test)
                st.session_state.selected_sample = {
                    "sample": X_test[selected_idx],
                    "class_label": y_test[selected_idx],
                    "binary_mask": None
                }
                st.session_state.selected_target = {
                    "sample": target_sample,
                    "class_label": target_class
                }
                st.success(f"Randomly selected target sample: Class {target_class}")
            else:
                filtered_X_test = { x: idx for idx, x in enumerate(range(len(X_test))) if y_test[idx] != y_test[selected_idx] }
                target_idx = st.selectbox(
                    "Manual selection of target sample:",
                    list(filtered_X_test.keys()),
                    format_func=lambda x: f" {x + 1} (Class {y_test[filtered_X_test[x]]})",
                    key="target_sample_selection"
                )
                if target_idx is not None:
                    target_sample = X_test[filtered_X_test[target_idx]]
                    target_class = y_test[filtered_X_test[target_idx]]
                    st.session_state.selected_target = {
                        "sample": target_sample,
                        "class_label": target_class
                    }
                    st.success(f"Selected target sample: Class {target_class}")

        st.info(f"""To select a region in the time series, click on the box select icon in the top right corner of the plot. 
                Then, click and drag to select a region.""")

        if selected_idx is not None:
            sample = X_test[selected_idx]
            binary_mask = self._render_interactive_plot_with_selection(sample, y_test[selected_idx])
            if binary_mask is not None:
                st.session_state.selected_sample = {
                    "sample": sample,
                    "class_label": y_test[selected_idx],
                    "binary_mask": binary_mask
                }
              
    def _render_interactive_plot_with_selection(self, sample, class_label):
        """
        Render the visualization for a selected sample and 
        allow the user to interactively select points in a popup.
        """
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(sample):
            sample = sample.detach().numpy()
        
        # Store original sample length for binary mask
        original_sample_length = len(sample)
        
        if "selected_target" in st.session_state and st.session_state.selected_target is not None:
            target_sample = st.session_state.selected_target['sample']
            target_class = st.session_state.selected_target['class_label']
            combined_df = pd.DataFrame({
                "time": list(range(len(sample))) + list(range(len(target_sample))),
                "value": list(sample) + list(target_sample),
                "series_type": ["Sample"] * len(sample) + ["Target"] * len(target_sample),
                "class": [class_label] * len(sample) + [target_class] * len(target_sample)
            })
        else:    
            combined_df = pd.DataFrame({
                "time": range(len(sample)),
                "value": sample,
                "class": class_label
            })

        fig = px.scatter(combined_df, x="time", y="value", 
                 color="series_type",
                 title="Time Series Sample",
                 labels={"time": "Time", "value": "Value", "class": "Class"},
                 template="plotly_white")
        
        # Add horizontal selection direction
        fig.update_layout(selectdirection='h')

        event = st.plotly_chart(fig, 
                                key="time serie", 
                                on_select="rerun")
        idx_selected = event['selection']['point_indices']
        # Create binary mask only for the original sample length
        binary_mask = np.zeros(original_sample_length)
        if idx_selected is not None and len(idx_selected) > 0:
            for idx in idx_selected:
                # Only consider indices that correspond to the original sample (not target)
                if 0 <= idx < original_sample_length:
                    binary_mask[idx] = 1
        else:
            return

        return binary_mask


    def _generate_local_counterfactuals_from_selection(self, sample, class_label, binary_mask, epochs, lr, beta):
        """
        Generate local counterfactuals from the selected region.
        
        :param sample: The original sample
        :param class_label: The class label of the sample
        :param binary_mask: Binary mask indicating selected points
        :param epochs: Number of epochs for optimization
        :param lr: Learning rate for optimization
        :param beta: Beta parameter for optimization
        """
        if not np.any(binary_mask):
            st.error("❌ No points selected in the binary mask. Please select a region to generate counterfactuals.")
            return
        
        try:
            if 'selected_target' in st.session_state and st.session_state.selected_target is not None:
                target_sample = st.session_state.selected_target['sample']
                target_class = st.session_state.selected_target['class_label']
                        
            # Convert sample to tensor if needed
            if not torch.is_tensor(sample):
                sample_tensor = torch.tensor(sample, dtype=torch.float32)
            else:
                sample_tensor = sample
            
            # Convert binary mask to tensor
            mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)

            # Get a random sample from an other class
            _, _, X_test, y_test = st.session_state.split_dataset
            # target_sample, target_class = get_target_from_base_class(class_label, y_test, X_test)

            if not torch.is_tensor(target_sample):
                target_sample = torch.tensor(target_sample, dtype=torch.float32)
            if not torch.is_tensor(target_class):
                target_class = torch.tensor(target_class, dtype=torch.int64)


            # Generate local counterfactuals
            local_counterfactual = counterfactual_local_generation(
                grsf_classifier=self.grsf_model,
                classifier=self.surrogate_model,
                target= target_sample,
                base=sample_tensor,
                base_label= class_label,
                binary_mask=mask_tensor,
                epochs=epochs,
                lr=lr,
                beta=beta
            )
            if local_counterfactual is None:
                st.error("❌ Failed to generate local counterfactual. Please check the selection and try again.")
                return None
            
            sample_np = sample_tensor.detach().numpy() if torch.is_tensor(sample_tensor) else sample_tensor
            target_sample_np = target_sample.detach().numpy() if torch.is_tensor(target_sample) else target_sample
            local_counterfactual = (local_counterfactual, 
                                    (target_sample_np, target_class), 
                                    (sample_np, class_label))

            return local_counterfactual
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération: {str(e)}")
        
            
class CounterfactualsAnalysisComponent:
    def __init__(self, counterfactuals_config: CounterfactualsConfig = None):
        """
        Initialize the component for counterfactual analysis.

        :param counterfactuals_config: Configuration for counterfactual generation (optional)
        """
        self.counterfactuals_config = counterfactuals_config

    def render(self, counterfactuals=None, grsf_classifier=None):
        """
        Render the UI for counterfactual analysis.
        
        :param counterfactuals: List of counterfactuals from session state
        :param grsf_classifier: GRSF classifier from session state
        """
        # Use passed parameters first, then session state, then config
        if counterfactuals is None:
            counterfactuals = st.session_state.get('counterfactuals_list', None)
        
        if grsf_classifier is None:
            grsf_classifier = st.session_state.get('trained_grsf_model', None)
            if grsf_classifier is None and self.counterfactuals_config is not None:
                grsf_classifier = self.counterfactuals_config.grsf_model
        
        # Check if we have the necessary data
        if counterfactuals is None or len(counterfactuals) == 0:
            st.warning("⚠️ No counterfactuals generated yet. Please generate them first.")
            return
        
        if grsf_classifier is None:
            st.error("❌ GRSF model not available for analysis.")
            return
        
        st.markdown("## 📊 Counterfactual Analysis")
        st.info(f"Analyzing {len(counterfactuals)} counterfactuals")
        
        # Tabs for different types of analysis
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Visual Analysis", "📏 Distance Analysis", "📈 Statistics", "🎯 Validity Check"])
        
        with tab1:
            self._render_visual_analysis(counterfactuals, grsf_classifier)
        
        with tab2:
            self._render_distance_analysis(counterfactuals, grsf_classifier)
        
        with tab3:
            self._render_statistics(counterfactuals, grsf_classifier)
        
        with tab4:
            self._render_validity_check(counterfactuals, grsf_classifier)

    def _render_visual_analysis(self, counterfactuals, grsf_classifier):
        """Render interactive time series visualizations of counterfactuals."""
        st.markdown("### 📈 Time Series Comparison")

        # Check if counterfactuals is valid
        if counterfactuals is None or len(counterfactuals) == 0:
            st.warning("⚠️ No counterfactuals available for visual analysis.")
            return

        # Select which counterfactual to visualize
        num_counterfactuals = len(counterfactuals)

        selected_idx = st.selectbox(
            "Select counterfactual to analyze:",
            range(num_counterfactuals),
            format_func=lambda x: f"Counterfactual {x + 1}",
            key="analysis_counterfactual_selection"
        )


        
        if selected_idx is not None:
            counterfactual, target, base = counterfactuals[selected_idx]
            
            # Convert tensors to numpy
            cf_np = counterfactual.detach().numpy() if torch.is_tensor(counterfactual) else counterfactual
            target_np = target[0].detach().numpy() if torch.is_tensor(target[0]) else target[0]
            base_np = base[0].detach().numpy() if torch.is_tensor(base[0]) else base[0]
            
            # Create interactive plot
            fig = go.Figure()
            
            time_steps = np.arange(len(cf_np))
            
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=base_np,
                mode='lines',
                name=f'Base (Class {base[1]})',
                line=dict(color='blue', dash='dash', width=2),
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=target_np,
                mode='lines',
                name=f'Target (Class {target[1]})',
                line=dict(color='red', width=2),
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=cf_np,
                mode='lines',
                name='Counterfactual',
                line=dict(color='green', dash='dashdot', width=3),
                opacity=0.9
            ))
            
            fig.update_layout(
                title=f'Counterfactual {selected_idx + 1} Analysis',
                xaxis_title='Time Steps',
                yaxis_title='Value',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

            validity = "✅" if grsf_classifier.predict(cf_np.reshape(1, -1)) != grsf_classifier.predict(base_np.reshape(1, -1)) else "❌"
            st.markdown(f"This counterfactual is valid: {validity}")
        
                # Grid view for multiple counterfactuals
        if st.checkbox("Show Grid View of All Counterfactuals", value=True, key="show_grid_view"):
            self._render_grid_view(counterfactuals, grsf_classifier)
        


    def _render_grid_view(self, counterfactuals, grsf_classifier):
        """Render a grid view of multiple counterfactuals."""
        st.markdown("### 📊 Grid View - All Counterfactuals")
        
        # Calculate grid dimensions
        num_counterfactuals = len(counterfactuals)
        cols = min(3, num_counterfactuals)
        rows = (num_counterfactuals + cols - 1) // cols
        
        # Create subplots
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[f'Counterfactual {i+1}' for i in range(num_counterfactuals)],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for i, (counterfactual, target, base) in enumerate(counterfactuals):
            if i >= rows * cols:
                break
                
            row = i // cols + 1
            col = i % cols + 1
            
            # Convert tensors to numpy
            cf_np = counterfactual.detach().numpy() if torch.is_tensor(counterfactual) else counterfactual
            target_np = target[0].detach().numpy() if torch.is_tensor(target[0]) else target[0]
            base_np = base[0].detach().numpy() if torch.is_tensor(base[0]) else base[0]
            
            time_steps = np.arange(len(cf_np))
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=time_steps, y=base_np, mode='lines', 
                          name=f'Base {i+1}', line=dict(color='blue', dash='dash'),
                          showlegend=(i == 0)), 
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(x=time_steps, y=target_np, mode='lines',
                          name=f'Target {i+1}', line=dict(color='red'),
                          showlegend=(i == 0)), 
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(x=time_steps, y=cf_np, mode='lines',
                          name=f'CF {i+1}', line=dict(color='green', dash='dashdot'),
                          showlegend=(i == 0)), 
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(x=time_steps, y=target_np, mode='lines',
                          name=f'Target {i+1}', line=dict(color='red'),
                          showlegend=(i == 0)), 
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(x=time_steps, y=cf_np, mode='lines',
                          name=f'CF {i+1}', line=dict(color='green', dash='dashdot'),
                          showlegend=(i == 0)), 
                row=row, col=col
            )

            if grsf_classifier.predict(cf_np.reshape(1, -1)) != grsf_classifier.predict(base_np.reshape(1, -1)):
                validity = "✅"
            else:
                validity = "❌"
            fig.add_annotation(
                text=f"Valid: {validity}",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12, color="black"),
                row=row, col=col
            )
        
        fig.update_layout(
            height=300 * rows,
            title_text="All Counterfactuals Overview",
            template='plotly_white'
        )
        
        

        st.plotly_chart(fig, use_container_width=True)


    def _render_distance_analysis(self, counterfactuals, grsf_classifier):
        """Render distance-based analysis with interactive charts."""
        st.markdown("### 📏 Distance Analysis")
        
        # Calculate distances
        distances_data = self._calculate_distances(counterfactuals, grsf_classifier)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Base distances visualization
            fig_base = go.Figure()
            
            fig_base.add_trace(go.Bar(
                x=['Euclidean', 'DTW'],
                y=[distances_data['base_euclidean_mean'], distances_data['base_dtw_mean']],
                name='Counterfactual vs Base',
                marker_color='lightblue',
                error_y=dict(
                    type='data',
                    array=[distances_data['base_euclidean_std'], distances_data['base_dtw_std']],
                    visible=True
                )
            ))
            
            fig_base.update_layout(
                title="Distance: Counterfactual ↔ Base",
                yaxis_title="Distance",
                template='plotly_white'
            )
            
            st.plotly_chart(fig_base, use_container_width=True)
        
        with col2:
            # Target distances visualization
            fig_target = go.Figure()
            
            fig_target.add_trace(go.Bar(
                x=['Euclidean', 'DTW'],
                y=[distances_data['target_euclidean_mean'], distances_data['target_dtw_mean']],
                name='Counterfactual vs Target',
                marker_color='lightcoral',
                error_y=dict(
                    type='data',
                    array=[distances_data['target_euclidean_std'], distances_data['target_dtw_std']],
                    visible=True
                )
            ))
            
            fig_target.update_layout(
                title="Distance: Counterfactual ↔ Target",
                yaxis_title="Distance",
                template='plotly_white'
            )
            
            st.plotly_chart(fig_target, use_container_width=True)
        
        # Distance distribution
        self._render_distance_distribution(distances_data)
        
        # PAA Analysis
        if st.checkbox("Show PAA (Piecewise Aggregate Approximation) Analysis", key="show_paa_analysis"):
            self._render_paa_analysis(counterfactuals)

    def _calculate_distances(self, counterfactuals, grsf_classifier):
        """Calculate various distance metrics between counterfactuals, bases, and targets."""
        base_euclidean = []
        base_dtw = []
        target_euclidean = []
        target_dtw = []
        sparsity_base = []
        sparsity_target = []
        valid_counterfactuals = 0

        if len(counterfactuals) <= 1:
            st.warning("⚠️ Not enough counterfactuals to calculate distances.")
            return {
                'base_euclidean_mean': 0,
                'base_euclidean_std': 0,
                'base_dtw_mean': 0,
                'base_dtw_std': 0,
                'target_euclidean_mean': 0,
                'target_euclidean_std': 0,
                'target_dtw_mean': 0,
                'target_dtw_std': 0,
                'base_euclidean_all': [],
                'base_dtw_all': [],
                'target_euclidean_all': [],
                'target_dtw_all': [],
                'sparsity_base': [],
                'sparsity_target': [],
                'valid_counterfactuals': valid_counterfactuals,
                'total_counterfactuals': len(counterfactuals)
            }
        
        for counterfactual, target, base in counterfactuals:
            # Convert to numpy
            cf_np = counterfactual.detach().numpy() if torch.is_tensor(counterfactual) else counterfactual
            target_np = target[0].detach().numpy() if torch.is_tensor(target[0]) else target[0]
            base_np = base[0].detach().numpy() if torch.is_tensor(base[0]) else base[0]
            
            # Base distances
            base_euclidean.append(np.linalg.norm(cf_np - base_np))
            base_dtw.append(wb_distance.dtw.dtw_distance(cf_np, base_np, r=1.0))
            
            # Target distances
            target_euclidean.append(np.linalg.norm(cf_np - target_np))
            target_dtw.append(wb_distance.dtw.dtw_distance(cf_np, target_np, r=1.0))
            
            # Sparsity
            sparsity_base.append(np.sum(cf_np != base_np) / cf_np.size * 100)
            sparsity_target.append(np.sum(cf_np != target_np) / cf_np.size * 100)
            
            # Validity check
            try:
                if grsf_classifier.predict(cf_np.reshape(1, -1)) != grsf_classifier.predict(base_np.reshape(1, -1)):
                    valid_counterfactuals += 1
            except:
                pass  # Skip if prediction fails
        
        return {
            'base_euclidean_mean': np.mean(base_euclidean),
            'base_euclidean_std': np.std(base_euclidean),
            'base_dtw_mean': np.mean(base_dtw),
            'base_dtw_std': np.std(base_dtw),
            'target_euclidean_mean': np.mean(target_euclidean),
            'target_euclidean_std': np.std(target_euclidean),
            'target_dtw_mean': np.mean(target_dtw),
            'target_dtw_std': np.std(target_dtw),
            'base_euclidean_all': base_euclidean,
            'base_dtw_all': base_dtw,
            'target_euclidean_all': target_euclidean,
            'target_dtw_all': target_dtw,
            'sparsity_base': sparsity_base,
            'sparsity_target': sparsity_target,
            'valid_counterfactuals': valid_counterfactuals,
            'total_counterfactuals': len(counterfactuals)
        }

    def _render_distance_distribution(self, distances_data):
        """Render distribution plots for distances."""
        st.markdown("#### 📊 Distance Distributions")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Base Euclidean', 'Base DTW', 'Target Euclidean', 'Target DTW'],
            vertical_spacing=0.12
        )
        
        # Base Euclidean
        fig.add_trace(
            go.Histogram(x=distances_data['base_euclidean_all'], name='Base Euclidean', 
                        marker_color='lightblue', showlegend=False),
            row=1, col=1
        )
        
        # Base DTW
        fig.add_trace(
            go.Histogram(x=distances_data['base_dtw_all'], name='Base DTW',
                        marker_color='lightgreen', showlegend=False),
            row=1, col=2
        )
        
        # Target Euclidean
        fig.add_trace(
            go.Histogram(x=distances_data['target_euclidean_all'], name='Target Euclidean',
                        marker_color='lightcoral', showlegend=False),
            row=2, col=1
        )
        
        # Target DTW
        fig.add_trace(
            go.Histogram(x=distances_data['target_dtw_all'], name='Target DTW',
                        marker_color='lightyellow', showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            height=500,
            title_text="Distance Distributions",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_paa_analysis(self, counterfactuals):
        """Render PAA (Piecewise Aggregate Approximation) analysis."""
        st.markdown("#### 🔍 PAA Analysis")
        
        n_segments = st.slider("Number of PAA segments", min_value=3, max_value=20, value=10, key="paa_segments_slider")
        
        try:
            paa = PAA(window_size=n_segments)
            paa_base_distances = []
            paa_target_distances = []
            
            for counterfactual, target, base in counterfactuals:
                # Convert to numpy
                cf_np = counterfactual.detach().numpy() if torch.is_tensor(counterfactual) else counterfactual
                target_np = target[0].detach().numpy() if torch.is_tensor(target[0]) else target[0]
                base_np = base[0].detach().numpy() if torch.is_tensor(base[0]) else base[0]
                
                # Apply PAA
                cf_paa = paa.transform(cf_np.reshape(1, -1))[0]
                target_paa = paa.transform(target_np.reshape(1, -1))[0]
                base_paa = paa.transform(base_np.reshape(1, -1))[0]
                
                paa_base_distances.append(np.linalg.norm(cf_paa - base_paa))
                paa_target_distances.append(np.linalg.norm(cf_paa - target_paa))
            
            # Create PAA comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['PAA Base Distance', 'PAA Target Distance'],
                y=[np.mean(paa_base_distances), np.mean(paa_target_distances)],
                marker_color=['lightblue', 'lightcoral'],
                error_y=dict(
                    type='data',
                    array=[np.std(paa_base_distances), np.std(paa_target_distances)],
                    visible=True
                )
            ))
            
            fig.update_layout(
                title=f"PAA Analysis ({n_segments} segments)",
                yaxis_title="Distance",
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in PAA analysis: {str(e)}")

    def _render_statistics(self, counterfactuals, grsf_classifier):
        """Render statistical summary of counterfactuals."""
        st.markdown("### 📈 Statistical Summary")
        if len(counterfactuals) <= 1:
            st.warning("⚠️ Not enough counterfactuals to calculate statistics.")
            return
        
        distances_data = self._calculate_distances(counterfactuals, grsf_classifier)
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Counterfactuals",
                distances_data['total_counterfactuals']
            )
        
        with col2:
            validity_rate = (distances_data['valid_counterfactuals'] / 
                           distances_data['total_counterfactuals'] * 100)
            st.metric(
                "Validity Rate",
                f"{validity_rate:.1f}%"
            )
        
        with col3:
            st.metric(
                "Avg Base Distance",
                f"{distances_data['base_euclidean_mean']:.3f}",
                f"±{distances_data['base_euclidean_std']:.3f}"
            )
        
        with col4:
            st.metric(
                "Avg Target Distance",
                f"{distances_data['target_euclidean_mean']:.3f}",
                f"±{distances_data['target_euclidean_std']:.3f}"
            )
        
        # Detailed statistics table
        st.markdown("#### 📋 Detailed Statistics")
        
        stats_df = pd.DataFrame({
            'Metric': ['Base Euclidean', 'Base DTW', 'Target Euclidean', 'Target DTW'],
            'Mean': [
                distances_data['base_euclidean_mean'],
                distances_data['base_dtw_mean'],
                distances_data['target_euclidean_mean'],
                distances_data['target_dtw_mean']
            ],
            'Std Dev': [
                distances_data['base_euclidean_std'],
                distances_data['base_dtw_std'],
                distances_data['target_euclidean_std'],
                distances_data['target_dtw_std']
            ],
            'Min': [
                min(distances_data['base_euclidean_all']),
                min(distances_data['base_dtw_all']),
                min(distances_data['target_euclidean_all']),
                min(distances_data['target_dtw_all'])
            ],
            'Max': [
                max(distances_data['base_euclidean_all']),
                max(distances_data['base_dtw_all']),
                max(distances_data['target_euclidean_all']),
                max(distances_data['target_dtw_all'])
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Sparsity analysis
        self._render_sparsity_analysis(distances_data)

    def _render_sparsity_analysis(self, distances_data):
        """Render sparsity analysis."""
        st.markdown("#### 🎯 Sparsity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_sparsity_base = np.mean(distances_data['sparsity_base'])
            st.metric(
                "Avg Sparsity vs Base",
                f"{avg_sparsity_base:.1f}%"
            )
        
        with col2:
            avg_sparsity_target = np.mean(distances_data['sparsity_target'])
            st.metric(
                "Avg Sparsity vs Target",
                f"{avg_sparsity_target:.1f}%"
            )
        
        # Sparsity distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=distances_data['sparsity_base'],
            name='Sparsity vs Base',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig.add_trace(go.Histogram(
            x=distances_data['sparsity_target'],
            name='Sparsity vs Target',
            marker_color='lightcoral',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Sparsity Distribution",
            xaxis_title="Sparsity (%)",
            yaxis_title="Frequency",
            barmode='overlay',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_validity_check(self, counterfactuals, grsf_classifier):
        """Render validity check analysis."""
        st.markdown("### 🎯 Validity Check")
        
        valid_predictions = 0
        invalid_predictions = 0
        prediction_errors = 0
        
        validity_details = []
        
        for i, (counterfactual, target, base) in enumerate(counterfactuals):
            try:
                cf_np = counterfactual.detach().numpy() if torch.is_tensor(counterfactual) else counterfactual
                base_np = base[0].detach().numpy() if torch.is_tensor(base[0]) else base[0]
                
                cf_pred = grsf_classifier.predict(cf_np.reshape(1, -1))[0]
                base_pred = grsf_classifier.predict(base_np.reshape(1, -1))[0]
                
                is_valid = cf_pred != base_pred
                
                if is_valid:
                    valid_predictions += 1
                else:
                    invalid_predictions += 1
                
                validity_details.append({
                    'Counterfactual': i + 1,
                    'Base Class': base[1],
                    'Target Class': target[1],
                    'CF Prediction': cf_pred,
                    'Base Prediction': base_pred,
                    'Valid': '✅' if is_valid else '❌'
                })
                
            except Exception as e:
                prediction_errors += 1
                validity_details.append({
                    'Counterfactual': i + 1,
                    'Base Class': base[1],
                    'Target Class': target[1],
                    'CF Prediction': 'Error',
                    'Base Prediction': 'Error',
                    'Valid': '❌'
                })
        
        # Summary metrics
        total = len(counterfactuals)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Valid", valid_predictions)
        
        with col2:
            st.metric("Invalid", invalid_predictions)
        
        with col3:
            st.metric("Errors", prediction_errors)
        
        with col4:
            validity_rate = (valid_predictions / total * 100) if total > 0 else 0
            st.metric("Validity Rate", f"{validity_rate:.1f}%")
        
        # Validity pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Valid', 'Invalid', 'Errors'],
            values=[valid_predictions, invalid_predictions, prediction_errors],
            marker_colors=['lightgreen', 'lightcoral', 'lightgray']
        )])
        
        fig.update_layout(
            title="Counterfactual Validity Distribution",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed validity table
        if st.checkbox("Show detailed validity table", key="show_validity_table"):
            validity_df = pd.DataFrame(validity_details)
            st.dataframe(validity_df, use_container_width=True)
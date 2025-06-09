import streamlit as st
import sys
import os
from typing import Dict, Any
from core.AppConfig import AppConfig
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
from counterfactual import counterfactual_batch_generation

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
        st.markdown("## 🔍 Counterfactuals generation parameters")
        nb_samples = st.number_input("Number of counterfactuals to generate", 
                                     min_value=1, 
                                     max_value=1000, 
                                     value=10, 
                                     step=1)
        st.session_state.generation_params = {
            "nb_samples": nb_samples,
        }
        
        st.divider()
        if st.button("Generate Counterfactuals"):
            try:
                st.session_state.counterfactuals_list = counterfactual_batch_generation(
                    grsf_classifier=self.grsf_model,
                    nn_classifier=self.surrogate_model,
                    split_dataset=self.split_dataset,
                    nb_samples=st.session_state.generation_params["nb_samples"],
                )
                st.success(f"Counterfactuals generated successfully! Total: {len(st.session_state.counterfactuals_list)}")
                return st.session_state.counterfactuals_list
            except Exception as e:
                error_msg = f"Error during counterfactual generation: {str(e)}"
                st.error(f"❌ {error_msg}")
                getLogger(__name__).error(error_msg)
                st.session_state.counterfactuals_list = None
                return None
            
        
            
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
            self._render_grid_view(counterfactuals)
        
        with tab2:
            self._render_distance_analysis(counterfactuals, grsf_classifier)
        
        with tab3:
            self._render_statistics(counterfactuals, grsf_classifier)
        
        with tab4:
            self._render_validity_check(counterfactuals, grsf_classifier)

    def _render_visual_analysis(self, counterfactuals):
        """Render interactive time series visualizations of counterfactuals."""
        st.markdown("### 📈 Time Series Comparison")
        # Grid view for multiple counterfactuals
        if st.button("Show all counterfactuals in grid view", value=True):
            self._render_grid_view(counterfactuals)
        else:
            # Select which counterfactual to visualize
            num_counterfactuals = len(counterfactuals)
            selected_idx = st.selectbox(
                "Select counterfactual to analyze:",
                range(num_counterfactuals),
                format_func=lambda x: f"Counterfactual {x + 1}"
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
                


    def _render_grid_view(self, counterfactuals):
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
        if st.checkbox("Show PAA (Piecewise Aggregate Approximation) Analysis"):
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
        
        n_segments = st.slider("Number of PAA segments", min_value=3, max_value=20, value=10)
        
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
        if st.checkbox("Show detailed validity table"):
            validity_df = pd.DataFrame(validity_details)
            st.dataframe(validity_df, use_container_width=True)
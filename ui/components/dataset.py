"""
Component for dataset visualization and exploration.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Optional, Dict, Any, List
import logging

from utils import getDataset

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetComponent:
    """
    Streamlit component for dataset visualization and exploration.
    
    This component provides a complete interface for analyzing time series datasets,
    including descriptive statistics, interactive visualizations, and class-based analysis.
    """
    
    def __init__(self, dataset_name: str) -> None:
        """
        Initialize the dataset component.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Raises:
            ValueError: If the dataset name is empty or None
        """
        if not dataset_name or not isinstance(dataset_name, str):
            raise ValueError("Dataset name must be a non-empty string")
            
        self.dataset_name = dataset_name.strip()
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize session variables if necessary."""
        if f"dataset_{self.dataset_name}_loaded" not in st.session_state:
            st.session_state[f"dataset_{self.dataset_name}_loaded"] = False
        
        if f"dataset_{self.dataset_name}_analysis_complete" not in st.session_state:
            st.session_state[f"dataset_{self.dataset_name}_analysis_complete"] = False
    
    @st.cache_data
    def _load_dataset(_self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the dataset with caching for performance optimization.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple containing the data (X) and labels (y)
            
        Raises:
            Exception: If dataset loading fails
        """
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            return getDataset(dataset_name)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise
    
    
    def _get_dataset_statistics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Calculate descriptive statistics of the dataset.
        
        Args:
            X: Time series data
            y: Class labels
            
        Returns:
            Dictionary containing the statistics
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        stats = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(unique_classes),
            'class_distribution': dict(zip(unique_classes, class_counts)),
            'data_range': (X.min(), X.max()),
            'mean_series_length': X.shape[1],
            'missing_values': np.isnan(X).sum(),
            'class_balance_ratio': class_counts.min() / class_counts.max() if class_counts.max() > 0 else 0
        }
        
        return stats
    
    def _display_dataset_info(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Display basic dataset information in a structured way.
        
        Args:
            X: Time series data
            y: Class labels
        """
        stats = self._get_dataset_statistics(X, y)
        
        # Header with title and description
        st.markdown("### 📊 Dataset Information")
        st.markdown(f"**Dataset:** `{self.dataset_name}`")
        
        # Main metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Samples",
                value=f"{stats['n_samples']:,}",
                help="Total number of samples in the dataset"
            )
        
        with col2:
            st.metric(
                label="Series Length",
                value=stats['n_features'],
                help="Number of time points per series"
            )
        
        with col3:
            st.metric(
                label="Classes",
                value=stats['n_classes'],
                help="Number of distinct classes"
            )
        
        with col4:
            balance_color = "normal" if stats['class_balance_ratio'] > 0.5 else "inverse"
            st.metric(
                label="Balance",
                value=f"{stats['class_balance_ratio']:.2f}",
                help="Balance ratio between classes (1.0 = perfectly balanced)",
                delta_color=balance_color
            )
        
        # Detailed information in an expander
        with st.expander("🔍 Technical Details"):
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**Value Range:**")
                st.code(f"Min: {stats['data_range'][0]:.3f}\nMax: {stats['data_range'][1]:.3f}")
                
                if stats['missing_values'] > 0:
                    st.warning(f"⚠️ {stats['missing_values']} missing values detected")
                else:
                    st.success("✅ No missing values")
            
            with detail_col2:
                st.markdown("**Class Distribution:**")
                for class_label, count in stats['class_distribution'].items():
                    percentage = (count / stats['n_samples']) * 100
                    st.markdown(f"- Class {class_label}: {count} ({percentage:.1f}%)")
        
        # Class distribution chart
        st.markdown("### 📈 Class Distribution")
        self._plot_class_distribution(stats['class_distribution'])
    
    def _plot_class_distribution(self, class_distribution: Dict[int, int]) -> None:
        """
        Create an interactive chart of the class distribution.
        
        Args:
            class_distribution: Dictionary {class: number_of_samples}
        """
        if not class_distribution:
            st.warning("No distribution data available")
            return
        
        # Data preparation for Plotly
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        total = sum(counts)
        percentages = [(count/total)*100 for count in counts]
        
        # Creating bar chart with Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f"Class {c}" for c in classes],
            y=counts,
            text=[f"{count}<br>({pct:.1f}%)" for count, pct in zip(counts, percentages)],
            textposition='auto',
            marker_color='lightblue',
            marker_line_color='darkblue',
            marker_line_width=1
        ))
        
        fig.update_layout(
            title="Sample distribution by class",
            xaxis_title="Classes",
            yaxis_title="Number of samples",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _visualize_class_samples(self, X: np.ndarray, y: np.ndarray, 
                                selected_class: int, max_samples: int = 5) -> None:
        """
        Visualize samples from the selected class.
        
        Args:
            X: Time series data
            y: Class labels
            selected_class: Class to visualize
            max_samples: Maximum number of samples to display
        """
        X_class = X[y == selected_class]
        
        if X_class.shape[0] == 0:
            st.warning(f"❌ No samples found for class {selected_class}")
            return
        
        num_samples = min(max_samples, X_class.shape[0])
        
        st.markdown(f"### 📊 Samples from Class {selected_class}")
        st.caption(f"Displaying {num_samples} samples out of {X_class.shape[0]} available")
        
        # Class statistics
        with st.expander("📈 Class statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{X_class.mean():.3f}")
            with col2:
                st.metric("Std Dev", f"{X_class.std():.3f}")
            with col3:
                st.metric("Median", f"{np.median(X_class):.3f}")
        
        # Sample visualization
        if num_samples <= 3:
            # Inline display for few samples
            cols = st.columns(num_samples)
            for i in range(num_samples):
                with cols[i]:
                    self._plot_single_timeseries(X_class[i], f"Sample {i+1}")
        else:
            # Using tabs for more samples
            tabs = st.tabs([f"Sample {i+1}" for i in range(num_samples)])
            
            for i, tab in enumerate(tabs):
                with tab:
                    self._plot_single_timeseries(X_class[i], f"Sample {i+1} - Class {selected_class}")
        

    
    def _plot_single_timeseries(self, series: np.ndarray, title: str) -> None:
        """
        Create a chart for a single time series.
        
        Args:
            series: Time series data
            title: Chart title
        """
        # Creating DataFrame for Plotly
        df = pd.DataFrame({
            'Time': range(len(series)),
            'Value': series
        })
        
        fig = px.line(
            df, 
            x='Time', 
            y='Value',
            title=title,
            markers=True
        )
        
        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Time points",
            yaxis_title="Value"
        )
        
        fig.update_traces(line=dict(width=2))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render(self) -> None:
        """Display the complete dataset component interface."""
        try:
            # Loading data with caching
            with st.spinner(f"🔄 Loading dataset '{self.dataset_name}'..."):
                X, y = self._load_dataset(self.dataset_name)
            
            # Mark as loaded in session
            st.session_state[f"dataset_{self.dataset_name}_loaded"] = True
            
            # Main layout with columns
            info_col, viz_col = st.columns([1, 2])
            
            with info_col:
                self._display_dataset_info(X, y)
                
                st.divider()
                
            
            with viz_col:
                # Selection section for visualization
                st.markdown("### 🎯 Class Visualization")
                
                unique_classes = np.unique(y)
                
                selected_class = st.selectbox(
                    "Class to visualize:",
                    options=unique_classes,
                    index=0,
                    help="Choose a class to display time series samples",
                    key=f"class_selector_{self.dataset_name}"
                )
                
                max_samples = st.slider(
                    "Max number of samples:",
                    min_value=1,
                    max_value=min(10, len(X[y == selected_class])),
                    value=min(5, len(X[y == selected_class])),
                    help="Maximum number of samples to display",
                    key=f"samples_slider_{self.dataset_name}"
                )
                
                # Button to trigger visualization
                if st.button(
                    "🚀 Visualize selected class", 
                    type="primary",
                    key=f"visualize_btn_{self.dataset_name}",
                    use_container_width=True
                ):
                    st.session_state[f"show_viz_{self.dataset_name}"] = True
                
                # Conditional display of visualization
                if st.session_state.get(f"show_viz_{self.dataset_name}", False):
                    with st.container():
                        self._visualize_class_samples(X, y, selected_class, max_samples)
                        
                        # Export options
                        st.divider()

                        # Button to clear visualization
                        if st.button(
                            "🧹 Clear visualization",
                            key=f"clear_viz_{self.dataset_name}"
                        ):
                            st.session_state[f"show_viz_{self.dataset_name}"] = False
                            st.rerun()
                else:
                    # Invitation message
                    st.info("👈 Select a class and click 'Visualize' to display samples")
                    
                    # Global statistics preview
                    stats = self._get_dataset_statistics(X, y)
                    
                    st.markdown("#### 📊 Statistical Overview")
                    st.markdown(f"""
                    - **Value range:** [{stats['data_range'][0]:.3f}, {stats['data_range'][1]:.3f}]
                    - **Class balance:** {stats['class_balance_ratio']:.2f}
                    - **Missing values:** {stats['missing_values']}
                    """)
            
        except Exception as e:
            error_msg = f"❌ Error loading dataset '{self.dataset_name}': {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)
            
            with st.expander("🛠️ Debug Information"):
                st.code(f"Requested dataset: {self.dataset_name}")
                st.code(f"Error type: {type(e).__name__}")
                st.code(f"Error message: {str(e)}")
            
            st.info("💡 Check that the dataset name is correct and that the dataset is available.")
            
        finally:
            # Mark analysis as complete
            st.session_state[f"dataset_{self.dataset_name}_analysis_complete"] = True

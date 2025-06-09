"""
Dataset exploration and analysis page.

This page allows users to select, load and analyze
different time series datasets available via wildboar.
"""
import streamlit as st
from components.dataset import DatasetComponent
import pandas as pd
import numpy as np
import logging
from typing import Optional

from utils import getDatasetNames

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPage:
    """
    Class for managing the dataset exploration page.
    
    This class encapsulates all dataset page logic,
    following the separation of concerns principle.
    """
    
    def __init__(self):
        """Initialize the dataset page."""
        self._configure_page()
        self._initialize_session_state()
    
    def _configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Dataset Explorer - GRSF",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if "dataset_loaded" not in st.session_state:
            st.session_state.dataset_loaded = False
        
        if "current_dataset" not in st.session_state:
            st.session_state.current_dataset = None
            
        if "dataset_analysis_started" not in st.session_state:
            st.session_state.dataset_analysis_started = False
    
    def _render_header(self) -> None:
        """Display the page header."""
        st.markdown("# 📊 Dataset Exploration")
        st.markdown("""
        Explore the available time series datasets for analysis 
        and counterfactual generation with GRSF.
        """)
        
        # Contextual information
        with st.expander("ℹ️ About datasets"):
            st.markdown("""
            This page uses the **wildboar** library which provides a collection 
            of standardized time series datasets for research and experimentation.
            """)
    
    def _render_dataset_selector(self) -> Optional[str]:
        """
        Display the dataset selector and return the selected dataset.
        
        Returns:
            Name of the selected dataset or None
        """
        st.markdown("## 🎯 Dataset Selection")
        
        try:
            # Loading available dataset names
            with st.spinner("🔄 Loading dataset list..."):
                available_datasets = getDatasetNames()
            
            if not available_datasets:
                st.error("❌ No datasets available.")
                return None
            
            # Selection interface
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_dataset = st.selectbox(
                    "Choose a dataset:",
                    options=available_datasets,
                    index=0,
                    key="dataset_selection",
                    help="Select a dataset from the list of available datasets"
                )
            
            with col2:
                st.metric(
                    "Available datasets",
                    len(available_datasets),
                    help="Total number of datasets in the collection"
                )
            
            # Information about the selected dataset
            if selected_dataset:
                st.info(f"📋 Selected dataset: **{selected_dataset}**")
                
                # Save to session
                st.session_state.current_dataset = selected_dataset
                
                return selected_dataset
                
        except Exception as e:
            error_msg = f"Error loading datasets: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)
            
            return None
    
    def _render_load_button(self, dataset_name: str) -> bool:
        """
        Display the load button and return True if clicked.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            True if the button was clicked
        """
        st.markdown("---")
        
        # Load button with state
        if not st.session_state.dataset_loaded or st.session_state.current_dataset != dataset_name:
            if st.button(
                f"🚀 Load and analyze '{dataset_name}'",
                type="primary",
                key="load_dataset",
                use_container_width=True,
                help=f"Load dataset {dataset_name} and display complete analysis"
            ):
                st.session_state.dataset_loaded = True
                st.session_state.dataset_analysis_started = True
                st.session_state.current_dataset = dataset_name
                return True
        else:
            # Dataset already loaded
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"✅ Dataset '{dataset_name}' loaded")
            
            with col2:
                if st.button("🔄 Reload", key="reload_dataset"):
                    st.session_state.dataset_loaded = False
                    st.session_state.dataset_analysis_started = False
                    st.rerun()
        
        return st.session_state.dataset_loaded and st.session_state.dataset_analysis_started
    
    def _render_dataset_analysis(self, dataset_name: str) -> None:
        """
        Display complete dataset analysis.
        
        Args:
            dataset_name: Name of the dataset to analyze
        """
        st.markdown("---")
        st.markdown("## 📈 Dataset Analysis")
        
        try:
            # Creating analysis component
            dataset_component = DatasetComponent(dataset_name)
            
            # Container for analysis
            with st.container():
                dataset_component.render()
                
            # Mark analysis as complete
            st.session_state.dataset_analysis_started = True
            
        except Exception as e:
            error_msg = f"Error during dataset analysis '{dataset_name}': {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)
            
            # Recovery options
            st.markdown("### 🛠️ Recovery Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Retry analysis"):
                    st.rerun()
            
            with col2:
                if st.button("🏠 Back to selection"):
                    st.session_state.dataset_loaded = False
                    st.session_state.dataset_analysis_started = False
                    st.rerun()
    
    def render(self) -> None:
        """Main method to display the complete page."""
        try:
            # Page header
            self._render_header()
            
            # Dataset selector
            selected_dataset = self._render_dataset_selector()
            
            if not selected_dataset:
                st.warning("⚠️ Please select a dataset to continue.")
                return
            
            # Load button
            should_analyze = self._render_load_button(selected_dataset)
            
            # Dataset analysis if requested
            if should_analyze:
                self._render_dataset_analysis(selected_dataset)
            
        except Exception as e:
            error_msg = f"Critical error in dataset page: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.critical(error_msg)
            
            # Error recovery interface
            st.markdown("### 🆘 Error Recovery")
            if st.button("🔄 Restart page"):
                # Complete state reset
                for key in list(st.session_state.keys()):
                    if key.startswith("dataset"):
                        del st.session_state[key]
                st.rerun()


def main():
    """Main entry point for the dataset page."""
    page = DatasetPage()
    page.render()


# Page execution
if __name__ == "__main__":
    main()
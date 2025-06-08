
"""
Configuration de l'application Streamlit pour la génération de contrefactuels.
"""
import streamlit as st
from typing import Dict, Any
import os


class AppConfig:
    """Configuration centralisée de l'application."""
    
    # Configuration de la page
    PAGE_CONFIG = {
        "page_title": "GRSF Counterfactual Generation",
        "page_icon": "🔬",
        "layout": "wide",
        "initial_sidebar_state": "collapsed",
    }
    
    # Thème et styles
    CUSTOM_CSS = """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 500;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    </style>
    """
    
    # Configuration des modèles
    MODEL_CONFIG = {
        "default_model": "grsf",
        "available_models": ["grsf", "baseline"],
        "model_params": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        }
    }
    
    # Configuration des données
    DATA_CONFIG = {
        "max_file_size": 200 * 1024 * 1024,  # 200MB
        "supported_formats": [".csv", ".xlsx", ".json"],
        "required_columns": ["features", "target"]
    }
    
    @staticmethod
    def configure_page():
        """Configure la page Streamlit avec les paramètres de base."""
        st.set_page_config(**AppConfig.PAGE_CONFIG)
        st.markdown(AppConfig.CUSTOM_CSS, unsafe_allow_html=True)
    
    @staticmethod
    def get_model_config() -> Dict[str, Any]:
        """Retourne la configuration des modèles."""
        return AppConfig.MODEL_CONFIG
    
    @staticmethod
    def get_data_config() -> Dict[str, Any]:
        """Retourne la configuration des données."""
        return AppConfig.DATA_CONFIG
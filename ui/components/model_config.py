"""
Component for model configuration.
"""
import streamlit as st
import sys
import os
from typing import Dict, Any
from core.AppConfig import AppConfig

import logging
from logging import getLogger

# Add the project root to the Python path to import gen.py
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now we can import gen.py from the project root
import gen
import models

# fOR GRSF
        # - n_shapelets: int, number of shapelets to be used
        # - metric: str, distance metric to be used
        # - min_shapelet_size: int, minimum size of the shapelets
        # - max_shapelet_size: int, maximum size of the shapelets
        # - alphas: list, list of alpha values to be used
    
distance_metrics = ["euclidean",
                    "normalized_euclidean",
                    "adtw",
                    "dtw",
                    "ddtw",
                    "wdtw",
                    "wddtw",
                    "lcss",
                    "wlcss",
                    "erp",
                    "edr",
                    "msm",
                    "twe",
                    "manhattan",
                    "minkowski",
                    "chebyshev",
                    "cosine",
                    "angular"]

class dnnUtils:
    def __init__(self):
        """
        Utility class for DNN configurations.
        """
        pass
    
    @staticmethod
    def get_available_models() -> list:
        """
        Get the list of available DNN models.
        
        Returns:
            list: List of model names
        """
        return models.listModels()

class dnnConfig:
    def __init__(self, model:gen.BaseSurrogateClassifier):
        self.params = model.get_params()
    
    def render(self) -> Dict[str, Any]:
        """
        Render the DNN configuration interface.
        
        Returns:
            Dict[str, Any]: Configuration of the DNN model
        """

        for param, value in self.params.items():
            st.markdown(f"**{param}**: (default) {value}")
            user_params = st.text_input(
                label=f"Set {param}",
                value=str(value),
                key=param,
                help=f"Set the value for {param} (default: {value})"
            )
        
        st.divider()
        st.markdown("### 🎞️ Counterfactual parameters")
        



class grsfConfig:
    """Configuration for GRSF model."""
    def __init__(self):
        """Initialize configuration for GRSF."""
        self.n_shapelets = 10
        self.metric = "euclidean"
        self.min_shapelet_size = 3
        self.max_shapelet_size = 10
        self.alphas = [0.1, 0.5, 1.0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return {
            "n_shapelets": self.n_shapelets,
            "metric": self.metric,
            "min_shapelet_size": self.min_shapelet_size,
            "max_shapelet_size": self.max_shapelet_size,
            "alphas": self.alphas
        }

    def train_model(self, dataset_name: str):
        """
        Train the GRSF model with current configuration.
        
        Args:
            dataset_name: Name of the dataset to train on
            
        Returns:
            Trained classifier and dataset splits
        """
        try:
            return gen.grsf(dataset_name, self.to_dict(), debug=True)
        except Exception as e:
            st.error(f"Error training GRSF model: {str(e)}")
            return None
    
    def save_config(self):
        """
        Display a popup to save the current configuration.
        """
        st.download_button(
            label="Save GRSF Configuration",
            data=str(self.to_dict()),
            file_name="grsf_config.json",
            mime="application/json",
            help="Download the current GRSF configuration as a JSON file"
        )
    
    def load_config(self):
        """
        Display a popup to load a configuration file.
        """
        uploaded_file = st.file_uploader(
            "Load GRSF Configuration",
            type=["json"],
            help="Upload a JSON file with GRSF configuration"
        )
        
        if uploaded_file is not None:
            try:
                config_data = uploaded_file.read().decode("utf-8")
                config_dict = eval(config_data)  # Use eval to convert string to dict
                self.n_shapelets = config_dict.get("n_shapelets", self.n_shapelets)
                self.metric = config_dict.get("metric", self.metric)
                self.min_shapelet_size = config_dict.get("min_shapelet_size", self.min_shapelet_size)
                self.max_shapelet_size = config_dict.get("max_shapelet_size", self.max_shapelet_size)
                self.alphas = config_dict.get("alphas", self.alphas)
                st.success("Configuration loaded successfully!")
            except Exception as e:
                st.error(f"Error loading configuration: {str(e)}")
    
    def evaluate_model(self, model, dataset):
        """
        Evaluate the GRSF model on the given dataset.
        
        Args:
            model: Trained GRSF model
            dataset: Dataset (X_train, y_train, X_test, y_test)
            
        Returns:
            Accuracy metric
        """
        try:
            _, _, X_test, y_test = dataset
            return gen.evaluate_grsf(model, X_test, y_test, debug=False)
        except Exception as e:
            st.error(f"Error evaluating GRSF model: {str(e)}")
            return None

    def render(self) -> None:
        """
        Display the GRSF model configuration interface.
        Returns:
            Dict[str, Any]: Configuration of the GRSF model
        """
        st.markdown("### ⚙️ GRSF Model Configuration")
        
        self.n_shapelets = st.number_input(
            "Number of shapelets",
            min_value=1,
            max_value=10000,
            value=self.n_shapelets,
            step=1
        )
        
        self.metric = st.selectbox(
            "Distance metric",
            options=distance_metrics,
            index=0
        )
        
        self.min_shapelet_size = st.number_input(
            "Minimum shapelet size",
            min_value=0,
            max_value=50,
            value=self.min_shapelet_size,
            step=1
        )
        
        self.max_shapelet_size = st.number_input(
            "Maximum shapelet size",
            min_value=self.min_shapelet_size,
            max_value=90,
            value=self.max_shapelet_size,
            step=1
        )
        
        # alpha values : triplet like (0.1, 1.0, 10.0)
        self.alphas=(0.1, 1.0, 10.0)
        return self.to_dict()

    
class ModelConfigComponent:
    """Composant pour configurer les paramètres du modèle."""
    
    def __init__(self):
        """Initialise le composant de configuration."""
        self.config = AppConfig.get_model_config()
    
    def render(self) -> Dict[str, Any]:
        """
        Affiche l'interface de configuration du modèle.
        
        Returns:
            Dict[str, Any]: Configuration du modèle sélectionnée
        """
        st.markdown("### ⚙️ Configuration du modèle")
        
        # Sélection du modèle
        selected_model = st.selectbox(
            "Modèle à utiliser",
            self.config["available_models"],
            index=0,
            help="Choisir le modèle pour la génération de contrefactuels"
        )
        
        st.markdown("#### Paramètres d'entraînement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.number_input(
                "Taux d'apprentissage",
                min_value=0.0001,
                max_value=0.1,
                value=self.config["model_params"]["learning_rate"],
                step=0.0001,
                format="%.4f"
            )
            
            epochs = st.number_input(
                "Nombre d'époques",
                min_value=10,
                max_value=1000,
                value=self.config["model_params"]["epochs"],
                step=10
            )
        
        with col2:
            batch_size = st.selectbox(
                "Taille de batch",
                [16, 32, 64, 128],
                index=1,  # Default to 32
                help="Taille du batch pour l'entraînement"
            )
            
            # Options avancées
            with st.expander("🔧 Options avancées"):
                early_stopping = st.checkbox(
                    "Arrêt précoce",
                    value=True,
                    help="Arrêter l'entraînement si aucune amélioration"
                )
                
                patience = st.number_input(
                    "Patience (arrêt précoce)",
                    min_value=5,
                    max_value=50,
                    value=10,
                    disabled=not early_stopping
                )
                
                validation_split = st.slider(
                    "Fraction de validation",
                    min_value=0.1,
                    max_value=0.3,
                    value=0.2,
                    step=0.05
                )
        
        # Configuration finale
        model_config = {
            "model_type": selected_model,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": early_stopping,
            "patience": patience,
            "validation_split": validation_split
        }
        
        # Affichage du résumé
        with st.expander("📋 Résumé de la configuration"):
            st.json(model_config)
        
        return model_config
    
    def save_config(self, config: Dict[str, Any]):
        """
        Sauvegarde la configuration dans la session.
        
        Args:
            config: Configuration à sauvegarder
        """
        st.session_state["model_config"] = config
    
    def load_config(self) -> Dict[str, Any]:
        """
        Charge la configuration depuis la session.
        
        Returns:
            Dict[str, Any]: Configuration chargée
        """
        return st.session_state.get("model_config", self.config)

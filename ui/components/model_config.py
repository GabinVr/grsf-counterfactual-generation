"""
Component for model configuration.
"""
import streamlit as st
import sys
import os
from typing import Dict, Any
from core.AppConfig import AppConfig
from code_editor import code_editor
import logging
from logging import getLogger

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import gen
import models
    
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

    @staticmethod
    def get_model_architecture(SurrogateClassifierArchitecture: gen.BaseSurrogateClassifier) -> str:
        modelsDict = models.listModels()
        print(f"DEBUG modelsDict: {modelsDict}, SurrogateClassifierArchitecture: {SurrogateClassifierArchitecture}")
        if isinstance(SurrogateClassifierArchitecture, str):
            SurrogateClassifierArchitecture = getattr(gen, SurrogateClassifierArchitecture, None)
            if SurrogateClassifierArchitecture is None:
                logging.error(f"Model {SurrogateClassifierArchitecture} not found in gen module.")
                return "Unknown Model"
        return [name for name, cls in modelsDict.items() if cls == SurrogateClassifierArchitecture][0] if SurrogateClassifierArchitecture in modelsDict.values() else "Unknown Model" 

    
class useDNN:
    """
    Class to use DNN models for counterfactual generation.
    """
    
    def __init__(self, model:gen.BaseSurrogateClassifier):
        """
        Setup the DNN model
        """
        self.model = model
    
    def getSetupModel(self, **kwargs) -> gen.BaseSurrogateClassifier:
        """
        Setup the DNN model with the given parameters.
        
        Args:
            **kwargs: Model parameters
        
        Returns:
            gen.BaseSurrogateClassifier: Configured DNN model
        """
        try:
            model_instance = self.model(**kwargs)
            return model_instance
        except Exception as e:
            st.error(f"Error configuring model {self.model.__name__}: {str(e)}")
            logging.error(f"Error configuring model {self.model.__name__}: {str(e)}")
            return None
    
class dnnConfig:
    def __init__(self, model:gen.BaseSurrogateClassifier):
        self.params = None
        self.model = model
    
    def set_params(self, sample_size: int = None, num_classes: int = None):
        """
        Set the parameters for the DNN model.
        
        Args:
            sample_size: Length of the input sequence
            num_classes: Number of output classes
        """
        self.params = self.model.get_params(sample_size=sample_size, num_classes=num_classes)
    
    def render(self) -> Dict[str, Any]:
        """
        Render the DNN configuration interface.
        """

        for param, desc in self.params.items():
        
            if desc["type"] == "int":
                value = st.number_input(
                    label=f"Set {param}",
                    min_value=desc.get("min", 0),
                    max_value=desc.get("max", 10000),
                    value=int(desc["default"]),
                    step=1,
                    help=desc.get("description", "")
                )
            elif desc["type"] == "float":
                value = st.number_input(
                    label=f"Set {param}",
                    min_value=desc.get("min", 0.0),
                    max_value=desc.get("max", 1.0),
                    value=float(desc["default"]),
                    step=0.01,
                    format="%.2f",
                    help=desc.get("description", "")
                )
            elif desc["type"] == "str":
                value = st.text_input(
                    label=f"Set {param}",
                    value=desc["default"],
                    help=desc.get("description", "")
                )
            elif desc["type"] == "bool":
                value = st.checkbox(
                    label=f"Set {param}",
                    value=desc["default"],
                    help=desc.get("description", "")
                )
            else:
                st.error(f"Unsupported parameter type for {param}: {desc['type']}")
                continue
            # Save the value from the user input
            st.session_state[param] = value

        # Setup the model with the parameters
        try:
            setup = useDNN(self.model)
            self.model = setup.getSetupModel(**{param: st.session_state[param] for param in self.params.keys()})
        except Exception as e:
            st.error(f"Error setting up model with parameters: {str(e)}")
            logging.error(f"Error setting up model with parameters: {str(e)}")
            return {}
    
    def train_model(self, split_dataset: tuple, epochs: int = 100, learning_rate: float = 0.001) -> gen.BaseSurrogateClassifier:
        """
        Train the DNN model with the given dataset.
        
        Args:
            split_dataset: Tuple containing (X_train, y_train, X_test, y_test)
            epochs: Number of training epochs
            learning_rate: Learning rate for the optimizer
        
        Returns:
            gen.BaseSurrogateClassifier: Trained DNN model
        """
        st.info("Printing training trace in the code editor below. You can copy it to your local machine for further analysis.")
        try:
            X_train, y_train, _, _ = split_dataset
            st.session_state["surrogate_training_progress"] = ""
            out =  self.model.train(X_train, y_train, epochs=epochs, lr=learning_rate, 
                                    training_callback=self._training_callback, debug=True)
            return out
        except Exception as e:
            st.error(f"Error training DNN model: {str(e)}")
            logging.error(f"Error training DNN model: {str(e)}")
            return None
    
    def _render_training_trace(self) -> None:
        """
        Render the training trace of the DNN model.
        """
        if "surrogate_training_progress" in st.session_state:
            code_editor(st.session_state["surrogate_training_progress"], 
                        height=7,
                        focus=False)
        return



    @staticmethod 
    def _training_callback(epoch: int, loss: float) -> None:
        """
        Callback function to display training progress.
        
        Args:
            epoch: Current epoch number
            loss: Loss value for the current epoch
        """
        st.session_state["surrogate_training_progress"] += f"Epoch {epoch + 1}: Loss = {loss:.4f}\n"
        return

    def evaluate_model(self, model: gen.BaseSurrogateClassifier, dataset: tuple) -> float:
        """
        Evaluate the DNN model on the given dataset.
        
        Args:
            model: Trained DNN model
            dataset: Dataset (X_train, y_train, X_test, y_test)
        
        Returns:
            float: Accuracy of the model on the test set
        """
        try:
            _, _, X_test, y_test = dataset
            return self.model.evaluate(X_test, y_test, debug=False)
        except Exception as e:
            st.error(f"Error evaluating DNN model: {str(e)}")
            logging.error(f"Error evaluating DNN model: {str(e)}")
            return None
        
class grsfConfig:
    """Configuration for GRSF model."""
    def __init__(self):
        """Initialize configuration for GRSF."""
        self.n_shapelets = 500
        self.metric = "euclidean"
        self.min_shapelet_size = 0.0
        self.max_shapelet_size = 1.0
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
            min_value=0.0,
            max_value=50.0,
            value=self.min_shapelet_size,
            step=0.1
        )
        
        self.max_shapelet_size = st.number_input(
            "Maximum shapelet size",
            min_value=self.min_shapelet_size,
            max_value=90.0,
            value=self.max_shapelet_size,
            step=0.1
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

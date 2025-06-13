import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional
from utils import getDatasetNames, getDataset
from components.model_config import grsfConfig, dnnConfig, dnnUtils
from components.couterfactuals_config import CounterfactualsConfig, CounterfactualsAnalysisComponent
from code_editor import code_editor

## Page with 4 tabs: 
# - Dataset selection
#     - Same as dataset page
# - Configuration & training of grsf model
#     - Simply expose the model configuration options with explanations
#     - training is done in the background when the user clicks "train"
#     - Display a progress bar or spinner during training
#     - Display the accuracy of the model after training
# - Choice of generation models and parameters
#     - Expose the generation parameters with explanations
#     - Display a button to launch the generation
#     - Display a progress bar or spinner during generation
# - Results display and analysis
#     - Display the generated counterfactuals with base and target classes/data

class GenerationPage:    
    def __init__(self):
        """Initialise la page de génération."""
        pass

    def render(self):

        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 Dataset selection",
            "⚙️ GRSF Model Configuration",
            "🎯 Generation Parameters",
            "📊 Results Display and Analysis"])
        
        with tab1:
            # Title and description of the tab
            st.markdown("### 📁 Dataset selection")

            datasets = getDatasetNames()
            selected_dataset = st.selectbox(
                "Sélectionner un dataset",
                datasets,
                help="Choose a dataset for counterfactual generation",
            )
            if selected_dataset:
                with st.spinner("Data loading..."):
                    dataset = getDataset(selected_dataset)
                    st.session_state['uploaded_data'] = dataset
                    st.session_state['dataset_name'] = selected_dataset
                    st.success(f"Dataset '{selected_dataset}' loaded successfully!")
                    # Store important information about the dataset in session state
                    st.session_state['sample_size'] = dataset[0].shape[1] # Length of the time series
                    st.session_state['num_classes'] = len(np.unique(dataset[1])) # Number of classes
                    st.divider()
                    st.markdown("Important information about the dataset:")
                    st.write(f"Sample size: {st.session_state['sample_size']}")
                    st.write(f"Number of classes: {st.session_state['num_classes']}")

        with tab2:
            # Only show this tab if data is uploaded
            if 'uploaded_data' not in st.session_state:
                st.warning("Please upload a dataset first in the previous tab.")
            else:
                # Display model configuration options
                model_config = grsfConfig()
                st.session_state['model_config'] = model_config.render()
                # Display the button to train the model
                st.divider()
                column_train, column_save, column_load = st.columns([2, 1, 1])
                with column_train:    
                    if st.button("🚀 Train GRSF Model", type="primary", use_container_width=True):
                        with st.spinner("Training GRSF model..."):
                            st.session_state['trained_grsf_model'], st.session_state['split_dataset'] = model_config.train_model(st.session_state['dataset_name'])
                        if 'trained_grsf_model' in st.session_state and st.session_state['trained_grsf_model'] != 1:
                            st.success("GRSF model trained successfully!")
                        else:
                            st.error("Failed to train GRSF model. Please check the configuration and dataset.")
                with column_save:
                    model_config.save_config()
                with column_load:
                    if st.button("📥 Load Model Configuration", type="secondary", use_container_width=True):
                        if 'model_config' in st.session_state:
                            st.session_state['model_config'].load_config()
                            st.success("Model configuration loaded successfully!")
                        else:
                            st.error("No model configuration found to load.")
                st.divider()
                if 'trained_grsf_model' in st.session_state:
                    st.markdown("#### 🚀 GRSF Model Accuracy")  
                    st.write(f"Accuracy: {model_config.evaluate_model(st.session_state['trained_grsf_model'], st.session_state['split_dataset']) * 100:.2f}%")
                
        with tab3:
            st.markdown("### 🎯 Generation parameters")
            st.markdown("Here you can choose the parameters, the model architecture, the loss function...")
            if 'trained_grsf_model' not in st.session_state:
                st.warning("Please train the GRSF model first in the previous tab.")
            else:

                st.divider()
                st.markdown("## 🛜 model selection")
                # let the user choose the model architecture and save it in the session state
                selected_model = st.selectbox(
                    "Select a surrogate model",
                    dnnUtils.get_available_models().keys(),
                    help="Choose the surrogate model architecture for counterfactual generation"
                )
                # Save the selected model in the session state
                st.session_state['selected_model'] = selected_model

                with st.expander("## 🛠 write your own ?"):
                    # Display the template for custom model (code in the file `template_custom_model.txt`)
                    st.markdown("You can fill in the code below to implement your own model architecture.")
                    try:
                        with open("ui/templates/template_custom_model.txt", "r") as file:
                            template_code = file.read()
                    except FileNotFoundError:
                        st.info("Template file not found. Please ensure the file exists in the correct path.")
                        template_code = ""
                    response_button = [{"name": "Finish"}]
                    response_dict = code_editor(template_code, buttons=response_button)
                    st.info(f"Custom model code updated: {response_dict}")
                    st.divider()
                st.divider()
                st.markdown("## ✨ Model parameters")
                # Todo - Add method in every model to give parameters needed
                dnn_model = dnnUtils.get_available_models()[st.session_state['selected_model']]
                dnn_model_conf = dnnConfig(dnn_model)
                dnn_model_conf.set_params(
                    sample_size=st.session_state['sample_size'],
                    num_classes=st.session_state['num_classes']
                )
                dnn_model_conf.render()
                st.divider()
                st.markdown("### Training parameters")
                # epochs 
                epochs = st.number_input(
                    "Number of epochs",
                    min_value=1,
                    max_value=1000,
                    value=100,
                    help="Number of epochs for training the surrogate model"
                )
                st.session_state['epochs'] = epochs
                # learning rate
                learning_rate = st.number_input(
                    "Learning rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.001,
                    step=0.0001,
                    help="Learning rate for training the surrogate model"
                )
                
                # Add a button to launch the training of the model
                if st.button("🚀 Train Surrogate Model", type="primary", use_container_width=True):
                    with st.spinner("Training surrogate model..."):
                        try:
                            st.info(f"split dataset: {len(st.session_state['split_dataset'])} ")

                            st.session_state['trained_surrogate_model'] = dnn_model_conf.train_model(st.session_state['split_dataset'], epochs=epochs, learning_rate=learning_rate)
                            st.success("Surrogate model trained successfully!")
                            st.divider()
                            accuracy = dnn_model_conf.evaluate_model(st.session_state['trained_surrogate_model'], st.session_state['split_dataset'])
                            st.markdown(f"#### 🚀 Surrogate Model Accuracy: {accuracy * 100:.2f}%")
                        except Exception as e:
                            st.error(f"Failed to train surrogate model: {str(e)}")
                            st.session_state['trained_surrogate_model'] = None
                if 'trained_surrogate_model' in st.session_state and st.session_state['trained_surrogate_model'] is not None:
                    # Initialize counterfactuals config if not exists
                    if 'counterfactuals_config' not in st.session_state:
                        counterfactuals_config = CounterfactualsConfig(
                            grsf_model=st.session_state['trained_grsf_model'],
                            surrogate_model=st.session_state['trained_surrogate_model'],
                            split_dataset=st.session_state['split_dataset']
                        )
                        st.session_state['counterfactuals_config'] = counterfactuals_config
                    else:
                        counterfactuals_config = st.session_state['counterfactuals_config']
                    
                    # Render counterfactuals generation UI
                    st.divider()
                    st.markdown("### 🎯 Generate Counterfactuals")
                    global_cf, local_cf = st.tabs(["🌍 Global", "📍 Local"])
                    with global_cf:
                        counterfactuals_config._render_global_counterfactuals()
                    with local_cf:
                        st.markdown("#### 📍 Local Counterfactuals Generation")
                        counterfactuals_config._render_local_counterfactuals()

        with tab4:
            # Get data from session state
            counterfactuals = st.session_state.get('counterfactuals_list', None)
            grsf_model = st.session_state.get('trained_grsf_model', None)
            
            # Create analysis component and pass data directly
            analysis_component = CounterfactualsAnalysisComponent(
                counterfactuals_config=st.session_state.get('counterfactuals_config', None)
            )
            
            # Pass the data directly to render method
            analysis_component.render(
                counterfactuals=counterfactuals,
                grsf_classifier=grsf_model
            )

def main():
    """Point d'entrée principal de l'application."""
    st.set_page_config(
        page_title="GRSF Counterfactual Generation",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialisation de la page de génération
    generation_page = GenerationPage()
    
    # Affichage de la page
    generation_page.render()

if __name__ == "__main__":
    main()
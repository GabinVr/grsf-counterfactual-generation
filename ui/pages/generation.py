import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import getDatasetNames, getDataset, getDistanceMetrics
from components.model_config import dnnConfig, dnnUtils
from model.dataset import DatasetObject
from model.grsf_model import GRSFModelObject    
from model.surrogate_model import SurrogateModelObject
from model.generation import LocalCounterfactualGeneratorObject
from components.graph import render_graph
import random
import pandas as pd

class GenerationPage:
    def __init__(self):
        self._configure_page()
        self._initialize_session_state()
        self._generationStateManager = None

    def _configure_page(self) -> None:
        st.set_page_config(
            page_title="Counterfactual Generation - GRSF",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    def _initialize_session_state(self) -> None:
        if "dataset" not in st.session_state:
            st.session_state.dataset = DatasetObject()
        if "grsf_model" not in st.session_state:
            st.session_state.grsf_model = GRSFModelObject()
        if "surrogate_model" not in st.session_state:
            st.session_state.surrogate_model = SurrogateModelObject()
        if "generation_parameters" not in st.session_state:
            st.session_state.generation_parameters = {}

    def render(self):
        dataset_tab, grsf_tab, surrogate_tab, local_tab, batch_tab = st.tabs([
            "Dataset",
            "GRSF",
            "Surrogate",
            "Local",
            "Batch"
        ])
        with dataset_tab:
            self._render_dataset_tab()
            if st.button("Rerun"):
                st.rerun()
        with grsf_tab:
            self._render_grsf_tab()
        with surrogate_tab:
            self._render_surrogate_tab()
        with local_tab:
            self._render_local_tab()
        with batch_tab:
            self._render_batch_tab()

    def _render_dataset_tab(self):
        st.markdown("### 📁 Dataset selection")
        
        datasets_names = getDatasetNames()
        selected_dataset = st.selectbox(
            "Select dataset",
            datasets_names,
            help="Choose a dataset for counterfactual generation",
            index=6,
        )

        
        if selected_dataset:
            # State management
            if st.session_state.dataset.has_changed(selected_dataset):
                st.session_state.dataset.set_dataset(selected_dataset)
                st.session_state.grsf_model.clear()
                st.session_state.surrogate_model.clear()
            
            st.session_state.dataset.load_dataset()
            st.divider()
            st.markdown(str(st.session_state.dataset))


    def _render_grsf_tab(self):
        st.markdown("### 🤖 GRSF model configuration")
        if st.session_state.dataset.is_empty():
            st.warning("Please select a dataset first.")
            return
        
        n_shapelets = st.number_input(
            "Number of shapelets",
            min_value=1,
            max_value=10000,
            value=500,
            step=1
        )
        
        metric = st.selectbox(
            "Distance metric",
            options=getDistanceMetrics(),
            index=0
        )
        
        min_shapelet_size = st.number_input(
            "Minimum shapelet size",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.1
        )
        
        max_shapelet_size = st.number_input(
            "Maximum shapelet size",
            min_value=0.0,
            max_value=90.0,
            value=1.0,
            step=0.1
        )
        if min_shapelet_size >= max_shapelet_size or min_shapelet_size < 0 or max_shapelet_size <= 0:
            st.error("Minimum shapelet size must be less than maximum shapelet size.")
            return
        
        # alpha values : triplet like (0.1, 1.0, 10.0)
        alphas=(0.1, 1.0, 10.0)

        parameters = {
            "n_shapelets": n_shapelets,
            "metric": metric,
            "min_shapelet_size": min_shapelet_size,
            "max_shapelet_size": max_shapelet_size,
            "alphas": alphas
        }


        if st.button("Train GRSF model"):
            if st.session_state.grsf_model.is_empty() or st.session_state.grsf_model.has_changed(parameters):
                st.session_state.grsf_model.set_parameters(parameters)
                st.session_state.surrogate_model.clear()
            
            st.session_state.grsf_model.set_dataset(st.session_state.dataset.get_dataset(), 
                                                    st.session_state.dataset.get_dataset_name())
            st.session_state.grsf_model.train()

        st.divider()
        st.markdown(str(st.session_state.grsf_model))
        # Save the grsf model in the session state

    def _render_surrogate_tab(self):
        st.markdown("### 🧠 Surrogate model configuration")
        if st.session_state.grsf_model.is_empty():
            st.warning("Please train a GRSF model first.")
            return
        
        surrogate_arch = st.selectbox(
            "Surrogate model architecture",
            options=dnnUtils.get_available_models().keys(),
            index=0
        )
        
        dnn_model = dnnUtils.get_available_models()[surrogate_arch]
        sample_size, num_classes = st.session_state.dataset.get_sample_size(), st.session_state.dataset.get_num_classes()
        available_params = dnn_model.get_params(sample_size=sample_size, num_classes=num_classes)

        parameters = self._render_model_parameters_sel(available_params)
        
        if st.button("Train Surrogate model"):
            if st.session_state.surrogate_model.is_empty() or st.session_state.surrogate_model.has_changed(parameters):
                st.session_state.surrogate_model.set_parameters(parameters)
            
            st.session_state.surrogate_model.set_grsf_model(st.session_state.grsf_model.get_model())
            st.session_state.surrogate_model.set_model(dnn_model)
            st.session_state.surrogate_model.set_split_dataset(st.session_state.grsf_model.get_split_dataset())
                                                     
            st.session_state.surrogate_model.train()

        st.divider()
        st.markdown(str(st.session_state.surrogate_model)) # Prints accuracy and training trace

    def _render_model_parameters_sel(self, available_params):
        parameters = {}
        
        for param_name, param_info in available_params.items():
            if param_info["type"] == "int":
                value = st.number_input(
                    param_info["description"],
                    min_value=param_info.get("min", 0),
                    max_value=param_info.get("max", 10000),
                    value=param_info["default"],
                    step=1,
                    key=param_name
                )
            elif param_info["type"] == "float":
                value = st.number_input(
                    param_info["description"],
                    min_value=param_info.get("min", 0.0),
                    max_value=param_info.get("max", 1.0),
                    value=param_info["default"],
                    step=0.01,
                    key=param_name
                )
            elif param_info["type"] == "str":
                value = st.text_input(
                    param_info["description"],
                    value=param_info["default"],
                    key=param_name
                )
            else:
                st.error(f"Unsupported parameter type: {param_info['type']}")
                continue
            parameters[param_name] = value
        return parameters

    def _render_local_tab(self):
        st.markdown("### 🏠 Local counterfactual generation")
        if st.session_state.surrogate_model.is_empty():
            st.warning("Please train a surrogate model first.")
            return
        
        # Initialize local counterfactual generator if not exists
        if "local_generator" not in st.session_state:
            st.session_state.local_generator = LocalCounterfactualGeneratorObject()
        st.session_state.local_generator.set_models(st.session_state.grsf_model.get_model(),
                                                    st.session_state.surrogate_model.get_model())
                                                    
        
        st.divider()
        st.markdown("#### Local generation parameters")
        epochs = st.number_input(
            "Number of epochs",
            min_value=1,
            max_value=1000,
            value=100,
            step=1,
            key="local_epochs"
        )
        learning_rate = st.text_input(
            "Learning rate",
            value="0.001",
            key="local_learning_rate"
        )
        if not learning_rate.replace(".", "").isnumeric():
            st.error("Learning rate must be a numeric value.")
            return
        learning_rate = float(learning_rate)

        beta = st.text_input(
            "Beta (regularization parameter)",
            value="0.1",
            key="local_beta"
        )
        if not beta.replace(".", "").isnumeric():
            st.error("Beta must be a numeric value between 0 and 1.")
            return
        beta = float(beta)
        if beta < 0 or beta > 1:
            st.error("Beta must be between 0 and 1.")
            return

        parameters = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "beta": beta
        }
        st.session_state.local_generator.set_parameters(parameters)

        st.divider()
        st.markdown("#### Select instance for counterfactual generation")
        self._render_sample_selection()
        self._render_visual_selection()

        st.divider()
        st.markdown("#### Generate counterfactual")
        
        # Check if samples are selected
        
        if st.button("Generate Counterfactual"):
            if st.session_state.local_generator.has_changed(parameters):
                st.session_state.local_generator.set_parameters(parameters)   
            
            # Generate counterfactual
            with st.spinner("Generating counterfactual..."):
                
                ret = st.session_state.local_generator.generate()
                if ret:
                    st.success("✅ Counterfactual generated successfully!")
                else:
                    st.warning("⚠️ No counterfactual generated. Check the parameters or samples.")
        st.divider()
        st.markdown(str(st.session_state.local_generator))
        if st.session_state.local_generator.has_counterfactual():
            st.markdown("#### Counterfactual Visualization")
            base_sample = st.session_state.local_generator.get_base_sample()
            target_sample = st.session_state.local_generator.get_target_sample()
            counterfactual = st.session_state.local_generator.get_counterfactual()
            
            if base_sample is not None and target_sample is not None and counterfactual is not None:
                combined_df = pd.DataFrame({
                    "time": list(range(len(base_sample))) + list(range(len(target_sample))) + list(range(len(counterfactual))),
                    "value": list(base_sample) + list(target_sample) + list(counterfactual),
                    "series_type": ["Base Sample"] * len(base_sample) + ["Target Sample"] * len(target_sample) + ["Counterfactual"] * len(counterfactual),
                    "class": [st.session_state.local_generator.get_base_class()] * len(base_sample) + [st.session_state.local_generator.get_target_class()] * len(target_sample) + [st.session_state.local_generator.get_target_class()] * len(counterfactual),
                })
                labels = {
                    "time": "Time",
                    "value": "Value",
                    "series_type": "Series Type",
                    "class": "Class"
                }
                render_graph(combined_df, labels, title="Counterfactual Generation Visualization")

    def _render_sample_selection(self):
        """
        Render the sample selection UI for local counterfactuals generation.
        """        
        _, _, X_test, y_test = st.session_state.grsf_model.get_split_dataset()
        
        if len(X_test) == 0:
            st.warning("⚠️ No test samples available for local counterfactuals generation.")
            return
        
        selected_idx = st.selectbox(
            "Select a base sample from the test set:",
            range(len(X_test)),
            format_func=lambda x: f"Sample {x + 1} (Class {y_test[x]})",
            key="local_sample_selection"
        )

        # let the user choose a target sample/class
        if selected_idx is not None:
            st.session_state.local_generator.set_base_sample(X_test[selected_idx], y_test[selected_idx])
            st.markdown("#### Select a target sample")
            rand_select = st.toggle("🎲 Randomly select a target sample", key="random_target_sample_toggle", value=True)
        
            if rand_select:
                if not st.session_state.local_generator.has_target_sample():
                    st.session_state.local_generator.select_random_target((None, None, X_test, y_test))
            else:
                different_class_indices = {i: i for i, label in enumerate(y_test) if label != y_test[selected_idx]}
                
                if different_class_indices:
                    target_idx = st.selectbox(
                        "Manual selection of target sample:",
                        list(different_class_indices.keys()),
                        format_func=lambda x: f"Sample {x + 1} (Class {y_test[x]})",
                        key="target_sample_selection"
                    )
                    
                    if target_idx is not None:
                        target_sample = X_test[target_idx]
                        target_class = y_test[target_idx]
                        
                        st.session_state.local_generator.set_target_sample(target_sample, target_class)
                else:
                    st.error("No samples from different classes available.")
    
    def _render_visual_selection(self) -> None:
        base_sample_np = st.session_state.local_generator.get_base_sample()
        target_sample_np = st.session_state.local_generator.get_target_sample()
        if base_sample_np is None or target_sample_np is None:
            st.warning("Please select a base and target sample first.")
            return
        combined_df = pd.DataFrame({
            "time": list(range(len(base_sample_np))) + list(range(len(target_sample_np))),
            "value": list(base_sample_np) + list(target_sample_np),
            "series_type": ["Base Sample"] * len(base_sample_np) + ["Target Sample"] * len(target_sample_np),
            "class": [st.session_state.local_generator.get_base_class()] * len(base_sample_np) + [st.session_state.local_generator.get_target_class()] * len(target_sample_np),
        })
        fig = px.scatter(combined_df, x="time", y="value", 
                         color="series_type",
                         title="Time Series Sample - Select points by dragging",
                         labels={"time": "Time", "value": "Value", "class": "Class"},
                         template="plotly_white")
        fig.update_layout(selectdirection='h',
                          dragmode='select')
        fig.update_traces(mode='lines+markers',
                          marker=dict(size=5, opacity=0.7),
                          line=dict(width=1.5))
        event = st.plotly_chart(fig, 
                                key="time_series_plot", 
                                on_select="rerun")

        idx_selected = event['selection']['point_indices']
        binary_mask = np.zeros(len(base_sample_np), dtype=int)
        if idx_selected is not None and len(idx_selected) > 0:
            for idx in idx_selected:
                # Only consider indices that correspond to the original sample (not target)
                if 0 <= idx < len(base_sample_np):
                    binary_mask[idx] = 1
        else:
            st.info("Click on the box select tool and drag to select points")
        st.session_state.local_generator.set_binary_mask(binary_mask)

    def _render_batch_tab(self):
        st.markdown("### 📦 Batch counterfactual generation")
        if st.session_state.surrogate_model.is_empty():
            st.warning("Please train a surrogate model first.")
            return
        
        # Initialize batch generator if not exists
        if "batch_generator" not in st.session_state:
            st.session_state.batch_generator = GblobalCounterfactualGeneratorObject()
def main():
    page = GenerationPage()
    page.render()
    

if __name__ == "__main__":
    main()
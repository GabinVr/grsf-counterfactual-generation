import streamlit as st
import pandas as pd
from typing import Optional
from utils import getDatasetNames, getDataset
from components.model_config import grsfConfig, dnnConfig, dnnUtils
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
                    with open("ui/templates/template_custom_model.txt", "r") as file:
                        template_code = file.read()
                    response_button = [{"name": "Finish"}]
                    response_dict = code_editor(template_code, buttons=response_button)
                    st.info(f"Custom model code updated: {response_dict}")
                    st.divider()
                st.divider()
                st.markdown("## ✨ Model parameters")
                # Todo - Add method in every model to give parameters needed
                dnn_model = dnnUtils.get_available_models()[st.session_state['selected_model']]
                dnn_model_conf = dnnConfig(dnn_model)
                dnn_model_conf.render()
                st.divider()
        with tab4:
            st.markdown("### 📊 Results Display and Analysis")
            st.markdown("Here you can visualize the generated counterfactuals and analyze their quality.")
            
    def _show_progress_status(self):
        """Affiche le statut de progression."""
        steps = [
            ("📁 Données", 'uploaded_data' in st.session_state),
            ("⚙️ Configuration", 'model_config' in st.session_state),
            ("🎯 Génération", 'generation_results' in st.session_state),
            ("📊 Analyse", False)  # Pour la page d'analyse
        ]
        
        cols = st.columns(len(steps))
        
        for i, (step_name, completed) in enumerate(steps):
            with cols[i]:
                if completed:
                    st.success(f"✅ {step_name}")
                else:
                    st.info(f"⏳ {step_name}")
    
    def _is_ready_for_generation(self) -> bool:
        """Vérifie si tous les prérequis sont remplis pour la génération."""
        return ('uploaded_data' in st.session_state and 
                'model_config' in st.session_state)
    
    def _render_generation_interface(self):
        """Affiche l'interface de génération."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Paramètres de génération
            st.markdown("#### Paramètres de génération")
            
            num_counteractuals = st.number_input(
                "Nombre de contrefactuels à générer",
                min_value=1,
                max_value=1000,
                value=10,
                help="Nombre d'exemples contrefactuels à créer"
            )
            
            target_class = st.selectbox(
                "Classe cible",
                ["Auto-détection", "Classe 0", "Classe 1"],
                help="Classe vers laquelle orienter les contrefactuels"
            )
            
            distance_threshold = st.slider(
                "Seuil de distance maximale",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Distance maximale autorisée par rapport à l'original"
            )
        
        with col2:
            # Options avancées
            st.markdown("#### Options avancées")
            
            preserve_features = st.multiselect(
                "Caractéristiques à préserver",
                st.session_state['uploaded_data'].columns.tolist(),
                help="Caractéristiques qui ne doivent pas être modifiées"
            )
            
            optimization_method = st.selectbox(
                "Méthode d'optimisation",
                ["Gradient Descent", "Genetic Algorithm", "Simulated Annealing"],
                help="Algorithme d'optimisation pour la génération"
            )
            
            random_seed = st.number_input(
                "Graine aléatoire",
                min_value=0,
                max_value=9999,
                value=42,
                help="Pour la reproductibilité des résultats"
            )
        
        # Bouton de génération
        st.markdown("---")
        
        if st.button("🚀 Lancer la génération", type="primary", use_container_width=True):
            self._run_generation({
                'num_counteractuals': num_counteractuals,
                'target_class': target_class,
                'distance_threshold': distance_threshold,
                'preserve_features': preserve_features,
                'optimization_method': optimization_method,
                'random_seed': random_seed
            })
    
    def _run_generation(self, generation_params: dict):
        """
        Lance la génération de contrefactuels.
        
        Args:
            generation_params: Paramètres de génération
        """
        with st.spinner("Génération en cours..."):
            # Ici, vous intégreriez votre logique de génération GRSF
            # Pour le moment, nous simulons avec des données factices
            
            progress_bar = st.progress(0)
            
            # Simulation de progression
            import time
            for i in range(100):
                time.sleep(0.01)  # Simulation du temps de traitement
                progress_bar.progress(i + 1)
            
            # Simulation de résultats
            original_data = st.session_state['uploaded_data']
            
            # Création de données contrefactuelles simulées
            # (à remplacer par votre logique réelle)
            counterfactual_data = self._simulate_counterfactuals(
                original_data, 
                generation_params['num_counteractuals']
            )
            
            # Sauvegarde des résultats
            st.session_state['generation_results'] = {
                'original_data': original_data,
                'counterfactual_data': counterfactual_data,
                'generation_params': generation_params,
                'metrics': self._calculate_metrics(original_data, counterfactual_data)
            }
            
            progress_bar.empty()
            st.success(f"✅ {generation_params['num_counteractuals']} contrefactuels générés avec succès!")
            st.rerun()
    
    def _simulate_counterfactuals(self, original_data: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        """
        Simule la génération de contrefactuels (à remplacer par la vraie logique).
        
        Args:
            original_data: Données originales
            num_samples: Nombre d'échantillons à générer
            
        Returns:
            pd.DataFrame: Données contrefactuelles simulées
        """
        import numpy as np
        
        # Sélection d'échantillons aléatoires
        sample_indices = np.random.choice(len(original_data), size=min(num_samples, len(original_data)), replace=False)
        counterfactual_data = original_data.iloc[sample_indices].copy()
        
        # Ajout de bruit gaussien pour simuler des modifications
        numeric_columns = counterfactual_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            noise = np.random.normal(0, counterfactual_data[col].std() * 0.1, len(counterfactual_data))
            counterfactual_data[col] += noise
        
        return counterfactual_data
    
    def _calculate_metrics(self, original: pd.DataFrame, counterfactual: pd.DataFrame) -> dict:
        """
        Calcule les métriques de qualité des contrefactuels.
        
        Args:
            original: Données originales
            counterfactual: Données contrefactuelles
            
        Returns:
            dict: Métriques calculées
        """
        import numpy as np
        
        # Métriques simulées (à remplacer par de vraies métriques)
        return {
            'average_distance': np.random.uniform(0.5, 1.5),
            'validity_rate': np.random.uniform(0.8, 1.0),
            'diversity_score': np.random.uniform(0.6, 0.9),
            'sparsity': np.random.uniform(0.3, 0.7)
        }
    
    def _render_results(self):
        """Affiche les résultats de génération."""
        results = st.session_state['generation_results']
        
        # Métriques de qualité
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = results['metrics']
        with col1:
            st.metric("Distance moyenne", f"{metrics['average_distance']:.2f}")
        with col2:
            st.metric("Taux de validité", f"{metrics['validity_rate']:.1%}")
        with col3:
            st.metric("Score de diversité", f"{metrics['diversity_score']:.2f}")
        with col4:
            st.metric("Parcimonie", f"{metrics['sparsity']:.2f}")
        
        # Aperçu des données générées
        st.markdown("#### Aperçu des contrefactuels générés")
        st.dataframe(results['counterfactual_data'].head(10))
        
        # Boutons d'action
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Analyser les résultats"):
                st.switch_page("pages/analysis.py")
        
        with col2:
            csv = results['counterfactual_data'].to_csv(index=False)
            st.download_button(
                "💾 Télécharger CSV",
                csv,
                "contrefactuels.csv",
                "text/csv"
            )
        
        with col3:
            if st.button("🔄 Nouvelle génération"):
                if 'generation_results' in st.session_state:
                    del st.session_state['generation_results']
                st.rerun()

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
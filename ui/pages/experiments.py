import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
from model.experimentLogger import ExperimentManager, ExperimentLoggerError


class ExperimentsPage:
    """
    Page for viewing and managing experiment results.
    """
    
    def __init__(self):
        self.experiment_manager = ExperimentManager()
        self._configure_page()
    
    def _configure_page(self) -> None:
        st.set_page_config(
            page_title="Experiments data - GRSF",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    
    def render(self):
        """Main render method for the experiments page."""
        st.title("📊 Experiment Results")
        st.markdown("View and analyze your counterfactual generation experiments.")
        
        overview_tab, details_tab, comparison_tab = st.tabs([
            "📋 Overview",
            "🔍 Details", 
            "⚖️ Comparison"
        ])
        
        with overview_tab:
            self._render_overview_tab()
        
        with details_tab:
            self._render_details_tab()
        
        with comparison_tab:
            self._render_comparison_tab()
    
    def _render_overview_tab(self):
        """Render the overview tab with experiment list."""
        st.markdown("### 📋 Experiments Overview")
        
        try:
            experiments = self.experiment_manager.list_experiments()
            
            if not experiments:
                st.info("No experiments found. Run some experiments in the Generation page to see results here.")
                return
            
            # Create overview dataframe
            overview_data = []
            for exp in experiments:
                overview_data.append({
                    'Name': exp['name'],
                    'Dataset': exp['dataset_name'],
                    'Created': exp['created_at'],
                    'GRSF Accuracy': exp['grsf_accuracy'],
                    'Surrogate Accuracy': exp['surrogate_accuracy'],
                    'Type': 'Local' if exp['has_local_generator'] else 'Batch',
                    'Size (KB)': round(exp['file_size'] / 1024, 1)
                })
            
            df = pd.DataFrame(overview_data)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Experiments", len(experiments))
            with col2:
                local_count = sum(1 for exp in experiments if exp['has_local_generator'])
                st.metric("Local Experiments", local_count)
            with col3:
                batch_count = sum(1 for exp in experiments if exp['has_batch_generator'])
                st.metric("Batch Experiments", batch_count)
            with col4:
                datasets_count = len(set(exp['dataset_name'] for exp in experiments))
                st.metric("Unique Datasets", datasets_count)
            
            # Display table
            st.markdown("### Experiment List")
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                dataset_filter = st.selectbox(
                    "Filter by Dataset",
                    options=['All'] + sorted(set(exp['dataset_name'] for exp in experiments)),
                    key="dataset_filter"
                )
            with col2:
                type_filter = st.selectbox(
                    "Filter by Type",
                    options=['All', 'Local', 'Batch'],
                    key="type_filter"
                )
            
            # Apply filters
            filtered_df = df.copy()
            if dataset_filter != 'All':
                filtered_df = filtered_df[filtered_df['Dataset'] == dataset_filter]
            if type_filter != 'All':
                filtered_df = filtered_df[filtered_df['Type'] == type_filter]
            
            # Display filtered table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
        except ExperimentLoggerError as e:
            st.error(f"Error loading experiments: {e}")
    
    def _render_overview_charts(self, experiments):
        """Render overview charts."""
        if not experiments:
            return
        
        st.markdown("### 📈 Experiment Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dataset distribution pie chart
            dataset_counts = {}
            for exp in experiments:
                dataset = exp['dataset_name']
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            
            fig_pie = px.pie(
                values=list(dataset_counts.values()),
                names=list(dataset_counts.keys()),
                title="Experiments by Dataset"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Accuracy comparison
            accuracy_data = []
            for exp in experiments:
                if exp['grsf_accuracy'] != 'N/A' and exp['surrogate_accuracy'] != 'N/A':
                    accuracy_data.append({
                        'Experiment': exp['name'][:15] + '...' if len(exp['name']) > 15 else exp['name'],
                        'GRSF': float(exp['grsf_accuracy']),
                        'Surrogate': float(exp['surrogate_accuracy']),
                        'Dataset': exp['dataset_name']
                    })
            
            if accuracy_data:
                acc_df = pd.DataFrame(accuracy_data)
                fig_acc = px.scatter(
                    acc_df, 
                    x='GRSF', 
                    y='Surrogate',
                    color='Dataset',
                    hover_data=['Experiment'],
                    title="GRSF vs Surrogate Accuracy"
                )
                fig_acc.add_shape(
                    type="line",
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(dash="dash", color="gray")
                )
                st.plotly_chart(fig_acc, use_container_width=True)
    
    def _render_details_tab(self):
        """Render the details tab for individual experiment analysis."""
        st.markdown("### 🔍 Experiment Details")
        
        try:
            experiments = self.experiment_manager.list_experiments()
            
            if not experiments:
                st.info("No experiments found.")
                return
            
            # Select experiment
            experiment_names = [exp['name'] for exp in experiments]
            selected_experiment = st.selectbox(
                "Select an experiment to analyze:",
                experiment_names,
                key="detail_experiment_select"
            )
            
            if selected_experiment:
                self._render_experiment_details(selected_experiment)
                
        except ExperimentLoggerError as e:
            st.error(f"Error loading experiments: {e}")
    
    def _render_experiment_details(self, experiment_name):
        """Render detailed view of a specific experiment."""
        try:
            experiment_data = self.experiment_manager.load_experiment(experiment_name)
            summary = self.experiment_manager.get_experiment_summary(experiment_name)
            
            # Basic info
            st.markdown(f"## Experiment: {experiment_name}")
            st.markdown(f"**Created:** {summary['created_at']}")
            
            # Tabs for different sections
            info_tab, data_tab, results_tab, export_tab = st.tabs([
                "ℹ️ Info", "📊 Data", "📈 Results", "💾 Export"
            ])
            
            with info_tab:
                self._render_experiment_info(summary)
            
            with data_tab:
                self._render_experiment_data(experiment_data)
            
            with results_tab:
                self._render_experiment_results(experiment_data)
            
            with export_tab:
                self._render_experiment_export(experiment_name, summary)
                
        except ExperimentLoggerError as e:
            st.error(f"Error loading experiment details: {e}")
    
    def _render_experiment_info(self, summary):
        """Render experiment information."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Dataset Information")
            dataset = summary['dataset']
            st.write(f"**Name:** {dataset.get('dataset_name', 'Unknown')}")
            st.write(f"**Samples:** {dataset.get('num_samples', 'N/A')}")
            st.write(f"**Features:** {dataset.get('sample_size', 'N/A')}")
            st.write(f"**Classes:** {dataset.get('num_classes', 'N/A')}")
            
            st.markdown("#### GRSF Model")
            grsf = summary['grsf_model']
            st.write(f"**Accuracy:** {grsf['accuracy']}")
            if grsf['parameters']:
                st.write("**Parameters:**")
                for key, value in grsf['parameters'].items():
                    st.write(f"- {key}: {value}")
        
        with col2:
            st.markdown("#### Surrogate Model")
            surrogate = summary['surrogate_model']
            st.write(f"**Architecture:** {surrogate['architecture']}")
            st.write(f"**Accuracy:** {surrogate['accuracy']}")
            if surrogate['parameters']:
                st.write("**Parameters:**")
                for key, value in surrogate['parameters'].items():
                    st.write(f"- {key}: {value}")
            
            st.markdown("#### Generation Info")
            st.write(f"**Type:** {summary['generation_type']}")
            
            if 'local_generation' in summary:
                local = summary['local_generation']
                st.write(f"**Generation Time:** {local['generation_time']} seconds")
                st.write(f"**Base Class:** {local['base_class']}")
                st.write(f"**Target Class:** {local['target_class']}")
                st.write(f"**Successfully Generated:** {local['is_generated']}")
            
            if 'batch_generation' in summary:
                batch = summary['batch_generation']
                batch_stats = batch.get('stats', {})
                valid_count = 0
                for stat in batch_stats:
                    if batch_stats[stat].get('valid', False):
                        valid_count += 1
                st.write(f"**Total Generated:** {len(batch_stats)}")
                st.write(f"**Valid Counterfactuals:** {valid_count/len(batch_stats) * 100:.2f}%")
    
    def _render_experiment_data(self, experiment_data):
        """Render experiment data visualizations."""
        if experiment_data.get('local_generator'):
            self._render_counterfactual_plot(experiment_data['local_generator'], title="Local Counterfactual Generation")
        elif experiment_data.get('batch_generator'):
            self._render_batch_generation_data(experiment_data['batch_generator'])
    
    def _render_counterfactual_plot(self, data, title="Counterfactual Generation Results"):
        st.markdown(f"#### {title}")
        
        # Check if we have the required data
        if not all(key in data for key in ['base_sample', 'target_sample', 'counterfactual']):
            st.warning("Missing data for visualization.")
            return
        
        # Create time series plot
        base_sample = data['base_sample']
        target_sample = data['target_sample']
        counterfactual = data['counterfactual']
        
        # Create combined dataframe
        time_points = list(range(len(base_sample)))
        
        df_combined = pd.DataFrame({
            'Time': time_points * 3,
            'Value': base_sample + target_sample + counterfactual,
            'Series': (['Base'] * len(base_sample) + 
                      ['Target'] * len(target_sample) + 
                      ['Counterfactual'] * len(counterfactual)),
            'Class': ([f"Class {data.get('base_class', '?')}"] * len(base_sample) +
                     [f"Class {data.get('target_class', '?')}"] * len(target_sample) +
                     [f"Class {data.get('target_class', '?')}"] * len(counterfactual))
        })
        
        fig = px.line(
            df_combined,
            x='Time',
            y='Value',
            color='Series',
            title="Counterfactual Generation Results",
            labels={'Value': 'Amplitude', 'Time': 'Time Steps'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Training progress
        if 'training_progress' in data and data['training_progress']:
            progress_text = data['training_progress']
            
            # Parse progress if it's in epoch format
            if "Epoch" in progress_text:
                lines = progress_text.strip().split('\n')
                epochs = []
                losses = []
                
                for line in lines:
                    if "Epoch" in line and "Loss" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            epoch_part = parts[0].split()[-1]
                            loss_part = parts[1].split("=")[-1].strip()
                            try:
                                epochs.append(int(epoch_part))
                                losses.append(float(loss_part))
                            except ValueError:
                                continue
                
                if epochs and losses:
                    fig_loss = px.line(
                        x=epochs,
                        y=losses,
                        title="Training Loss Over Time",
                        labels={'x': 'Epoch', 'y': 'Loss'}
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                else:
                    st.text_area("Training Log", progress_text, height=200)
            else:
                st.text_area("Training Log", progress_text, height=200)
        
        if data.get('stats'):
            validity = data['stats'].get('valid', None)
            if validity is not None:
                st.markdown(f"This counterfactual is {'valid ✅' if validity else 'invalid ❌'}.")

            
    
    def _render_batch_generation_data(self, batch_generator_data):
        """Render batch generation data."""
        st.markdown("#### Batch Counterfactual Generation")

        # Display general statistics
        if 'stats' in batch_generator_data:
            stats = batch_generator_data['stats']
            
            # Display statistics
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                
                if 'success_rate' in stats:
                    with col1:
                        st.metric("Success Rate", f"{stats['success_rate']:.2%}")
                
                if 'avg_generation_time' in stats:
                    with col2:
                        st.metric("Avg Generation Time", f"{stats['avg_generation_time']:.3f}s")
                
                if 'total_generated' in stats:
                    with col3:
                        st.metric("Total Generated", stats['total_generated'])
                
                # Generation time
                gen_time = batch_generator_data.get('generation_time', 0)
                with col4:
                    st.metric("Total Generation Time", f"{gen_time:.3f}s")
        
        # Display counterfactuals and loss curves by pair

        for id in range(len(batch_generator_data.get('counterfactuals', []))):
            cf_data = self._convert_batch_to_plot_format(batch_generator_data, id)
            self._render_counterfactual_plot(cf_data, title=f"Counterfactual #{id+1}")
            
        # Display additional batch statistics
        if 'parameters' in batch_generator_data:
            st.markdown("### Generation Parameters")
            params = batch_generator_data['parameters']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Epochs", params.get('epochs', 'N/A'))
            with col2:
                st.metric("Learning Rate", params.get('learning_rate', 'N/A'))
            with col3:
                st.metric("Beta", params.get('beta', 'N/A'))
    
    def _is_valid_counterfactual_data(self, cf_data):
        """Check if counterfactual data has the required structure."""
        if not isinstance(cf_data, list) or len(cf_data) == 0:
            return False
        
        # Check if it's a time series (list of numbers)
        try:
            # Verify it's a list of numbers
            for val in cf_data:
                float(val)
            return True
        except (ValueError, TypeError):
            return False
    
    def _convert_batch_to_plot_format(self, batch_generator_data, id):
        """Convert batch counterfactual data to format expected by _render_counterfactual_plot."""
        # let's unpack the triplet
        counterfactual, target, base = batch_generator_data['counterfactuals'][id]
        target_data, target_class = target
        base_data, base_class = base

        converted_data = {
            'base_sample': base_data,
            'target_sample': target_data,
            'counterfactual': counterfactual,
            'base_class': base_class,
            'target_class': target_class,
            'training_progress': batch_generator_data.get('training_progress', [])[id],
            'stats': batch_generator_data.get('stats', {}).get(str(id), {})
        }
        
        return converted_data
    
    def _render_batch_overview(self, counterfactuals, batch_data):
        """Render overview of batch counterfactuals."""
        st.markdown("#### Batch Overview")
        
        # Statistics about counterfactuals
        valid_counterfactuals = [cf for cf in counterfactuals if self._is_valid_counterfactual_data(cf)]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valid Counterfactuals", len(valid_counterfactuals))
        with col2:
            st.metric("Invalid/Empty", len(counterfactuals) - len(valid_counterfactuals))
        with col3:
            if valid_counterfactuals:
                avg_length = sum(len(cf) for cf in valid_counterfactuals) / len(valid_counterfactuals)
                st.metric("Avg Length", f"{avg_length:.1f}")
        
        # Show distribution of counterfactual lengths
        if valid_counterfactuals:
            lengths = [len(cf) for cf in valid_counterfactuals]
            
            fig_hist = px.histogram(
                x=lengths,
                nbins=min(20, len(set(lengths))),
                title="Distribution of Counterfactual Lengths"
            )
            fig_hist.update_layout(
                xaxis_title="Length (Time Steps)",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Show first few counterfactuals in a grid
            st.markdown("#### Sample Counterfactuals")
            sample_size = min(6, len(valid_counterfactuals))
            
            cols = st.columns(3)
            for i in range(sample_size):
                with cols[i % 3]:
                    cf_data = valid_counterfactuals[i]
                    
                    # Create mini plot
                    fig_mini = px.line(
                        x=list(range(len(cf_data))),
                        y=cf_data,
                        title=f"CF #{i+1}"
                    )
                    fig_mini.update_layout(height=200, showlegend=False)
                    st.plotly_chart(fig_mini, use_container_width=True)
    
    def _render_batch_comparison(self, counterfactuals, batch_data):
        """Render comparison view of counterfactuals."""
        st.markdown("#### Counterfactuals Comparison")
        
        valid_counterfactuals = [cf for cf in counterfactuals if self._is_valid_counterfactual_data(cf)]
        
        if len(valid_counterfactuals) < 2:
            st.warning("Need at least 2 valid counterfactuals for comparison.")
            return
        
        # Allow user to select which counterfactuals to compare
        max_compare = min(5, len(valid_counterfactuals))
        
        selected_indices = st.multiselect(
            "Select counterfactuals to compare (max 5):",
            range(len(valid_counterfactuals)),
            default=list(range(min(3, len(valid_counterfactuals)))),
            format_func=lambda x: f"Counterfactual {x+1}",
            key="batch_comparison_select"
        )
        
        if selected_indices:
            # Create comparison plot
            comparison_data = []
            colors = px.colors.qualitative.Set3[:len(selected_indices)]
            
            for i, idx in enumerate(selected_indices):
                cf_data = valid_counterfactuals[idx]
                time_points = list(range(len(cf_data)))
                
                for t, val in enumerate(cf_data):
                    comparison_data.append({
                        'Time': t,
                        'Value': val,
                        'Counterfactual': f"CF #{idx+1}",
                        'Index': idx
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                
                fig_comparison = px.line(
                    df_comparison,
                    x='Time',
                    y='Value',
                    color='Counterfactual',
                    title="Counterfactuals Comparison",
                    labels={'Value': 'Amplitude', 'Time': 'Time Steps'}
                )
                fig_comparison.update_layout(height=500)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Show statistics comparison
                st.markdown("#### Comparison Statistics")
                stats_data = []
                for idx in selected_indices:
                    cf_data = valid_counterfactuals[idx]
                    stats_data.append({
                        'Counterfactual': f"CF #{idx+1}",
                        'Length': len(cf_data),
                        'Min Value': min(cf_data),
                        'Max Value': max(cf_data),
                        'Mean': sum(cf_data) / len(cf_data),
                        'Std Dev': (sum((x - sum(cf_data) / len(cf_data))**2 for x in cf_data) / len(cf_data))**0.5
                    })
                
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    def _render_batch_training_curves(self, training_progress):
        """Render training curves for batch generation."""
        if not training_progress or not isinstance(training_progress, list):
            st.warning("No training progress data available.")
            return
        
        # Parse training progress for each counterfactual
        all_curves_data = []
        
        for cf_idx, progress_text in enumerate(training_progress[:5]):  # Limit to first 5 for clarity
            if isinstance(progress_text, str) and "Epoch" in progress_text:
                lines = progress_text.strip().split('\n')
                epochs = []
                losses = []
                
                for line in lines:
                    if "Epoch" in line and "Loss" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            epoch_part = parts[0].split()[-1]
                            loss_part = parts[1].split("=")[-1].strip()
                            try:
                                epochs.append(int(epoch_part))
                                losses.append(float(loss_part))
                            except ValueError:
                                continue
                
                # Add to combined data
                for epoch, loss in zip(epochs, losses):
                    all_curves_data.append({
                        'Epoch': epoch,
                        'Loss': loss,
                        'Counterfactual': f"CF #{cf_idx+1}"
                    })
        
        if all_curves_data:
            df_curves = pd.DataFrame(all_curves_data)
            
            # Create training curves plot
            fig_curves = px.line(
                df_curves,
                x='Epoch',
                y='Loss',
                color='Counterfactual',
                title="Training Loss Curves for Batch Generation",
                labels={'Loss': 'Training Loss', 'Epoch': 'Epoch'}
            )
            fig_curves.update_layout(height=500)
            st.plotly_chart(fig_curves, use_container_width=True)
            
            # Show final losses comparison
            final_losses = df_curves.groupby('Counterfactual')['Loss'].last().reset_index()
            final_losses['Loss'] = final_losses['Loss'].round(4)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Final Training Losses")
                st.dataframe(final_losses, hide_index=True)
            
            with col2:
                # Bar chart of final losses
                fig_bar = px.bar(
                    final_losses,
                    x='Counterfactual',
                    y='Loss',
                    title="Final Training Losses"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Could not parse training progress data.")
        
        
    
    def _render_experiment_results(self, experiment_data):
        """Render experiment results and analysis."""
        st.markdown("#### Results Analysis")
        
        # Model accuracies comparison
        grsf_acc = experiment_data.get('grsf_model', {}).get('accuracy', 0)
        surrogate_acc = experiment_data.get('surrogate_model', {}).get('accuracy', 0)
        
        if grsf_acc and surrogate_acc:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "GRSF Model Accuracy",
                    f"{grsf_acc:.3f}",
                    delta=None
                )
            
            with col2:
                delta = surrogate_acc - grsf_acc
                st.metric(
                    "Surrogate Model Accuracy",
                    f"{surrogate_acc:.3f}",
                    delta=f"{delta:+.3f}"
                )
        
        # Generation success metrics
        if experiment_data.get('local_generator'):
            local_gen = experiment_data['local_generator']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Generation Success",
                    "✅ Yes" if local_gen.get('is_generated', False) else "❌ No"
                )
            
            with col2:
                gen_time = local_gen.get('generation_time', 0)
                st.metric("Generation Time", f"{gen_time:.3f}s")
            
            with col3:
                base_class = local_gen.get('base_class', '?')
                target_class = local_gen.get('target_class', '?')
                st.metric("Class Change", f"{base_class} → {target_class}")
    
    def _render_experiment_export(self, experiment_name, summary):
        """Render experiment export options."""
        st.markdown("#### Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Export Summary as Text", key=f"export_text_{experiment_name}"):
                summary_text = self.experiment_manager.export_experiment_summary(experiment_name)
                st.download_button(
                    label="💾 Download Summary",
                    data=summary_text,
                    file_name=f"{experiment_name}_summary.md",
                    mime="text/markdown",
                    key=f"download_summary_{experiment_name}"
                )
        
        with col2:
            if st.button("📊 Export Data as JSON", key=f"export_json_{experiment_name}"):
                try:
                    experiment_data = self.experiment_manager.load_experiment(experiment_name)
                    json_str = json.dumps(experiment_data, indent=2)
                    st.download_button(
                        label="💾 Download JSON",
                        data=json_str,
                        file_name=f"{experiment_name}.json",
                        mime="application/json",
                        key=f"download_json_{experiment_name}"
                    )
                except Exception as e:
                    st.error(f"Error preparing export: {e}")
        
        # Display summary preview
        with st.expander("📋 Summary Preview"):
            summary_text = self.experiment_manager.export_experiment_summary(experiment_name)
            st.markdown(summary_text)
    
    def _render_comparison_tab(self):
        """Render the comparison tab."""
        st.markdown("### ⚖️ Experiment Comparison")
        
        try:
            experiments = self.experiment_manager.list_experiments()
            
            if len(experiments) < 2:
                st.info("Need at least 2 experiments for comparison.")
                return
            
            # Select experiments to compare
            experiment_names = [exp['name'] for exp in experiments]
            
            col1, col2 = st.columns(2)
            with col1:
                exp1 = st.selectbox("First Experiment", experiment_names, key="comp_exp1")
            with col2:
                exp2 = st.selectbox("Second Experiment", experiment_names, key="comp_exp2")
            
            if exp1 and exp2 and exp1 != exp2:
                self._render_experiment_comparison(exp1, exp2)
                
        except ExperimentLoggerError as e:
            st.error(f"Error loading experiments for comparison: {e}")
    
    def _render_experiment_comparison(self, exp1_name, exp2_name):
        """Render comparison between two experiments."""
        try:
            summary1 = self.experiment_manager.get_experiment_summary(exp1_name)
            summary2 = self.experiment_manager.get_experiment_summary(exp2_name)
            
            # Comparison table
            comparison_data = {
                'Metric': [],
                exp1_name: [],
                exp2_name: []
            }
            
            # Add metrics
            metrics = [
                ('Dataset', 'dataset', 'dataset_name'),
                ('GRSF Accuracy', 'grsf_model', 'accuracy'),
                ('Surrogate Accuracy', 'surrogate_model', 'accuracy'),
                ('Generation Type', 'generation_type', None)
            ]
            
            for metric_name, key1, key2 in metrics:
                comparison_data['Metric'].append(metric_name)
                
                if key2:
                    val1 = summary1.get(key1, {}).get(key2, 'N/A') if key1 in summary1 else 'N/A'
                    val2 = summary2.get(key1, {}).get(key2, 'N/A') if key1 in summary2 else 'N/A'
                else:
                    val1 = summary1.get(key1, 'N/A')
                    val2 = summary2.get(key1, 'N/A')
                
                comparison_data[exp1_name].append(val1)
                comparison_data[exp2_name].append(val2)
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
        except ExperimentLoggerError as e:
            st.error(f"Error comparing experiments: {e}")

def main():
    """Main function to run the experiments page."""
    page = ExperimentsPage()
    page.render()


if __name__ == "__main__":
    main()
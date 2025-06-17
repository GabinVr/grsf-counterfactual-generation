import streamlit as st
import pandas as pd
import numpy as np
import wildboar.datasets as wb_datasets
import torch

import plotly.express as px
import plotly.graph_objects as go

def render_interactive_plot_with_selection(sample, target, class_label, target_class):
    """
    Render the visualization for a selected sample and 
    allow the user to interactively select points.
    """
    
    # Convert to numpy if it's a tensor
    if hasattr(sample, 'detach'):  # Check if it's a tensor
        sample = sample.detach().numpy()
    
    # Convert to numpy if it's a tensor
    if hasattr(target, 'detach'):  # Check if it's a tensor
        target = target.detach().numpy()
    
    # Create combined dataframe with both sample and target
    sample_df = pd.DataFrame({
        "time": list(range(len(sample))) + list(range(len(target))),
        "value": list(sample) + list(target),
        "series_type": ["Sample"] * len(sample) + ["Target"] * len(target),
        "class": [class_label] * len(sample) + [target_class] * len(target)
    })

    fig = px.scatter(sample_df, x="time", y="value", 
             color="series_type",
             title="Time Series Sample - Select points by dragging",
             labels={"time": "Time", "value": "Value", "class": "Class"},
             template="plotly_white")
    

    
    # Add horizontal selection direction
    fig.update_layout(selectdirection='h')

    # Display the plot and capture selection events
    event = st.plotly_chart(fig, 
                            key="time_series_plot", 
                            on_select="rerun")
    
    # Process selected points
    st.info(f"DEBUG: {event}")
    idx_selected = event['selection']['point_indices']
    binary_mask = np.zeros(len(sample), dtype=bool)
    
    if idx_selected is not None and len(idx_selected) > 0:
        for idx in idx_selected:
            if 0 <= idx < len(sample):
                binary_mask[idx] = True
        
        st.success(f"Selected {len(idx_selected)} points")
        st.write(f"Selected indices: {idx_selected}")
    else:
        st.info("Click on the box select tool and drag to select points")
    
    return binary_mask

# Example usage
def main():
    st.title("Interactive Time Series Plot Selection")
    
    # Dataset selection
    st.sidebar.title("Dataset Configuration")
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ["ECG200", "Trace", "Coffee", "Gun_Point", "Lightning2"]
    )
    
    # Load dataset
    try:
        X, y = wb_datasets.load_dataset(dataset_name)
        st.sidebar.success(f"Dataset {dataset_name} loaded successfully!")
        st.sidebar.write(f"Shape: {X.shape}")
        st.sidebar.write(f"Classes: {np.unique(y)}")
        
        # Sample selection
        sample_idx = st.sidebar.slider(
            "Select Sample Index",
            0, len(X) - 1, 0
        )
        
        # Get selected sample
        sample_data = X[sample_idx]
        class_label = y[sample_idx]

        target_data = X[sample_idx+1] if sample_idx + 1 < len(X) else X[sample_idx-1]
        target_label = y[sample_idx+1] if sample_idx + 1 < len(X) else y[sample_idx-1]
        
        st.write(f"**Dataset:** {dataset_name}")
        st.write(f"**Sample Index:** {sample_idx}")
        st.write(f"**Class Label:** {class_label}")
        st.write(f"**Target Sample Index:** {sample_idx + 1 if sample_idx + 1 < len(X) else sample_idx - 1}")
        st.write(f"**Target Class Label:** {target_label}")
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Fallback to generated data
        np.random.seed(42)
        sample_data = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
        class_label = 1
        st.warning("Using generated sample data")
    
    st.info("Use the box select tool in the plot toolbar to select a region")
    
    # Render the interactive plot
    selected_mask = render_interactive_plot_with_selection(sample_data, 
                                                            target_data, 
                                                            class_label, 
                                                            target_label)
    
    # Display results
    if np.any(selected_mask):
        st.subheader("Selection Results")
        st.write(f"Number of selected points: {np.sum(selected_mask)}")
        
        # Show selected region
        selected_indices = np.where(selected_mask)[0]
        if len(selected_indices) > 0:
            st.write(f"Selected range: {selected_indices[0]} to {selected_indices[-1]}")
            
            # Plot with highlighted selection
            fig_result = go.Figure()
            
            # All points
            fig_result.add_trace(go.Scatter(
                x=np.arange(len(sample_data)),
                y=sample_data,
                mode='lines+markers',
                name='Time Series',
                marker=dict(color='blue', size=4)
            ))
            
            # Selected points
            fig_result.add_trace(go.Scatter(
                x=selected_indices,
                y=sample_data[selected_mask],
                mode='markers',
                name='Selected Points',
                marker=dict(color='red', size=8)
            ))
            
            fig_result.update_layout(
                title="Time Series with Selected Region",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_white"
            )
            
            st.plotly_chart(fig_result, use_container_width=True)

if __name__ == "__main__":
    main()
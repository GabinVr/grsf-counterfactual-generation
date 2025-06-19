import streamlit as st
import plotly.express as px
from typing import Dict
import random
import hashlib
def render_graph(df,
                 labels: Dict[str, str],
                 title: str = "Graph",
                 ) -> None:
    """
    Render a graph using Plotly and display it in Streamlit.
    
    Args:
        graph: The Plotly graph object to render.
        df: DataFrame containing the data for the graph.
        title: Title of the graph.
    """
    unique_plot_id = f"graph_{hashlib.md5(str(random.random()).encode()).hexdigest()}"
    fig = px.scatter(df, x="time", y="value", 
                         color="series_type",
                         title=title,
                         labels=labels,
                         template="plotly_white")
    fig.update_traces(mode='lines+markers',
                        marker=dict(size=5, opacity=0.7),
                        line=dict(width=1.5))
    st.plotly_chart(fig,
                    on_select="rerun",
                    key=unique_plot_id,)

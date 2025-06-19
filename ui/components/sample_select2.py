from __future__ import annotations

"""SampleSelectComponent
=========================

Composant Streamlit permettant :
    • l'affichage d'un échantillon et de sa cible (ainsi qu'une contrefactuelle facultative)
    • la sélection horizontale de points directement sur le graphique Plotly
    • la génération du masque binaire correspondant aux points sélectionnés

Installation :
    pip install streamlit streamlit-plotly-events plotly

Lancement rapide en console :
    streamlit run sample_select_component.py

"""

from typing import Union, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from streamlit_plotly_events import plotly_events

__all__ = ["SampleSelectComponent", "example_usage"]

_TensorLike = Union[pd.Series, np.ndarray, torch.Tensor]


def _to_numpy(arr: _TensorLike) -> np.ndarray:  # pragma: no cover
    """Convertit torch, numpy ou pandas.Series en np.ndarray."""
    if isinstance(arr, pd.Series):
        return arr.values.astype(float)
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(float)
    if isinstance(arr, np.ndarray):
        return arr.astype(float)
    raise TypeError(
        "Type non supporté. Utilisez pd.Series, np.ndarray ou torch.Tensor."
    )


class SampleSelectComponent:
    """Composant interactif de visualisation & sélection de séries temporelles."""

    def __init__(
        self,
        sample: _TensorLike,
        target: _TensorLike,
        sample_class: str,
        target_class: str,
    ) -> None:
        self.sample = _to_numpy(sample)
        self.target = _to_numpy(target)

        if self.sample.shape != self.target.shape:
            raise ValueError("sample et target doivent avoir la même longueur.")

        self.class_label: str = sample_class
        self.target_class: str = target_class

        self.counterfactual: Optional[np.ndarray] = None
        self.selected_points: Optional[List[int]] = None
        self.binary_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Interface Streamlit
    # ------------------------------------------------------------------

    def display(self, key: str = "sample_select") -> None:
        """Affiche le graphique interactif dans l'app Streamlit."""
        fig = self._build_figure()

        selection = plotly_events(
            fig,
            select_event=True,
            click_event=False,
            override_height=400,
            key=key,
        )

        if selection:
            idx = sorted({int(round(ev["x"])) for ev in selection})
            self.selected_points = idx

            self.binary_mask = np.zeros_like(self.sample, dtype=int)
            self.binary_mask[idx] = 1

            st.info(f"{len(idx)} points sélectionnés sur {len(self.sample)}")
            st.write("Indices sélectionnés :", idx)

    # ------------------------------------------------------------------
    # API programmatique
    # ------------------------------------------------------------------

    def add_counterfactual(self, cf: _TensorLike) -> None:
        """Ajoute/modifie la série contrefactuelle."""
        cf_np = _to_numpy(cf)
        if cf_np.shape != self.sample.shape:
            raise ValueError("counterfactual doit avoir la même longueur que sample.")
        self.counterfactual = cf_np

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_figure(self) -> go.Figure:
        x = np.arange(self.sample.size)
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=self.sample, mode="lines", name=self.class_label))
        fig.add_trace(go.Scatter(x=x, y=self.target, mode="lines", name=self.target_class))

        if self.counterfactual is not None:
            fig.add_trace(
                go.Scatter(x=x, y=self.counterfactual, mode="lines", name="counterfactual")
            )

        fig.update_layout(
            dragmode="select",
            selectdirection="h",
            margin=dict(l=40, r=20, t=40, b=40),
            hovermode="x unified",
        )
        return fig


# ----------------------------------------------------------------------
# Example usage function ------------------------------------------------
# ----------------------------------------------------------------------

def example_usage() -> None:
    """Demo Streamlit complète du composant.

    Exécutez : ``streamlit run sample_select_component.py``
    """

    import numpy as np  # noqa: F811 – re‑import pour clarté dans la fonction

    # --- Données factices ---
    sample = np.sin(np.linspace(0, 8 * np.pi, 500))
    target = sample + np.random.normal(0, 0.1, 500)

    # --- Instanciation du composant ---
    selector = SampleSelectComponent(sample, target, "sample", "target")

    # --- Interface principale ---
    st.title("Demo SampleSelectComponent")
    selector.display()

    if st.button("Afficher une contrefactuelle"):
        cf = sample * 0.2  # Exemple de contrefactuelle
        selector.add_counterfactual(cf)

    # --- Affichage du masque ---
    if selector.binary_mask is not None:
        st.subheader("Masque binaire")
        st.write(selector.binary_mask)




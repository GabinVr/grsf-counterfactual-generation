#!/bin/bash

# Script de lancement pour l'application GRSF Counterfactual Generation
# Usage: ./run_app.sh

echo "🔬 Lancement de l'application GRSF Counterfactual Generation..."

# Lancement de l'application
echo "🚀 Démarrage de l'application Streamlit..."
echo "🌐 L'application sera disponible sur : http://localhost:8501"
echo "🛑 Appuyez sur Ctrl+C pour arrêter l'application"
echo ""

# Changement vers le dossier ui et lancement
cd ui && streamlit run main.py

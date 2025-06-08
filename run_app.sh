#!/bin/bash

# Script de lancement pour l'application GRSF Counterfactual Generation
# Usage: ./run_app.sh

echo "🔬 Lancement de l'application GRSF Counterfactual Generation..."

# Vérification de l'environnement virtuel
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Aucun environnement virtuel détecté."
    echo "💡 Recommandation : Activez votre environnement virtuel avec :"
    echo "   source .venv/bin/activate"
    echo ""
    read -p "Continuer quand même ? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Lancement de l'application
echo "🚀 Démarrage de l'application Streamlit..."
echo "🌐 L'application sera disponible sur : http://localhost:8501"
echo "🛑 Appuyez sur Ctrl+C pour arrêter l'application"
echo ""

# Changement vers le dossier ui et lancement
cd ui && streamlit run main.py

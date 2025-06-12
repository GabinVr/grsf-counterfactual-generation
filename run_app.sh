#!/bin/bash

# Launch script for the GRSF Counterfactual Generation application
# Usage: ./run_app.sh

echo "🔬 Launching GRSF Counterfactual Generation application..."
echo "🚀 Starting Streamlit App ..."
echo "🌐 Available @ : http://localhost:8501"
echo "🛑 press Ctrl+C to escape"
echo ""

streamlit run ui/main.py

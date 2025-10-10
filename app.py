# app.py
import streamlit as st
from pathlib import Path


from src.utils.st_utils import initialize_models
from src.utils.runs_utils import find_runs_in_dir
from src.st_sections import (
    page_live_prediction,
    side_bar,
    page_metrics_and_config,
    page_compare_models,
    page_home,
)

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Food101 Mini Demo", layout="wide")

# ----------------------------
# Load runs
# ----------------------------
models_root = Path("selected_models")
runs = find_runs_in_dir(models_root)

if not runs:
    st.error("No model runs found in 'selected_models'.")
    st.stop()

# Sort runs by test accuracy
runs.sort(
    key=lambda x: x["info"].get("test_metrics", {}).get("accuracy", 0), reverse=True
)

# Run sorted names
run_names = [r["run_path"].name for r in runs]

# Maping of runs by
run_map = {name: r for name, r in zip(run_names, runs)}

# Initialize all models in "loaded_models"
initialize_models(run_map)

pages = [
    "🏠 Home",
    "📷 Live Prediction",
    "📊 Metrics and Configuration",
    "📈 Compare All Models",
]

# ----------------------------
# Sidebar navigation
# ----------------------------
page = side_bar(pages)


# ----------------------------
# Render pages
# ----------------------------

if page == "🏠 Home":
    page_home(run_names=run_names)

elif page == "📷 Live Prediction":

    page_live_prediction(run_names=run_names)

elif page == "📊 Metrics and Configuration":

    page_metrics_and_config(run_map=run_map, run_names=run_names)


elif page == "📈 Compare All Models":

    page_compare_models(runs)

import streamlit as st
import torch
from src.utils.model_utils import load_model, get_model_transforms
import os

def load_selected_model(run):
    model_info = run["info"]
    hyperparameters = model_info.get("hyperparameters", {})
    class_names = model_info.get("class_names", [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)

    model = load_model(
        state_dict_path=run["state_dict"],
        model_name=hyperparameters["model_name"],
        num_classes=num_classes,
        version=hyperparameters.get("version"),
        device=device,
    )
    model.eval()

    transform = get_model_transforms(
        model_name=hyperparameters["model_name"],
        version=hyperparameters.get("version"),
        augmentation=None,
    )

    return model, transform, class_names


def initialize_models(run_map):
    """Load all models once and initialize session_state."""

    if "loaded_models" not in st.session_state:
        st.session_state.loaded_models = {}
        st.session_state.model_transforms = {}
        st.session_state.model_classnames = {}
        st.session_state.model_name = {}
        st.session_state.model_info = {}  # metrics, hyperparams, etc.
        st.session_state.num_params = {}
        st.session_state.size = {}
        st.session_state.state_dict_path={}

        # Seleccionar primer modelo por defecto
        st.session_state.selected_display = list(run_map.keys())[0]

        for name, run in run_map.items():
            model, transform, class_names = load_selected_model(run=run)
            st.session_state.loaded_models[name] = model
            st.session_state.model_transforms[name] = transform
            st.session_state.model_classnames[name] = class_names
            st.session_state.model_name[name] = name
            st.session_state.model_info[name] = run["info"]
            st.session_state.num_params[name] = sum(p.numel() for p in model.parameters())
            st.session_state.state_dict_path[name] = run["state_dict"]
            st.session_state.size[name] = os.path.getsize(run["state_dict"]) / 1024**2


            


def get_selected_model_info():
    """Return model, transform, class_names, and info of the selected model."""
    key = st.session_state.selected_display
    model = st.session_state.loaded_models[key]
    transform = st.session_state.model_transforms[key]
    class_names = st.session_state.model_classnames[key]
    info = st.session_state.model_info[key]
    return model, transform, class_names, info

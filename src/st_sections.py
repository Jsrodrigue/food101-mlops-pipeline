# src/st_sections.py
import pandas as pd
import streamlit as st
from PIL import Image
from torch import cuda
import matplotlib.pyplot as plt
import seaborn as sns

from src.predictions import predict_image
from src.utils.render_utils import (
    render_prediction_card,
    render_probability_bars,
    render_confusion_matrix,
)
import numpy as np


# ----------------------------------------
#                 SIDE BAR
# ----------------------------------------


def side_bar(pages):
    with st.sidebar:
        st.subheader("Navigation")

        # --- Initialize session state for current page ---
        if "page" not in st.session_state:
            st.session_state.page = "🏠 Home"

        for page_name in pages:
            is_active = st.session_state.page == page_name
            color = "#0d3b66" if is_active else "#555"
            font_weight = "700" if is_active else "500"

            if st.button(page_name, key=page_name):
                st.session_state.page = page_name

        # --- CSS only for sidebar navigation buttons ---
        st.markdown(
            f"""
          <style>
          [data-testid="stSidebar"] div.stButton > button[kind] {{
              background-color: transparent;
              color: {color};
              font-weight: {font_weight};
              border: none;
              padding: 0.2rem 0;
              text-align: left;
              cursor: pointer;
              font-size: 16px;
              transition: color 0.2s ease, font-weight 0.2s ease;
          }}
          [data-testid="stSidebar"] div.stButton > button[kind]:hover {{
              color: #0d3b66;
              font-weight: 600;
          }}
          </style>
      """,
            unsafe_allow_html=True,
        )

    # Handle query parameter for page
    query_params = st.query_params
    if "page" in query_params:
        st.session_state.page = query_params["page"][0]

    page = st.session_state.page

    return page


# ------------------------------------------
#            HOME
# -----------------------------------------


def page_home(run_names=None):
    run_names = run_names or []

    # Main title
    st.markdown(
        """
        <h1 style='text-align:center; font-size: 2.5em; margin-bottom: 0.2em;'>🍔 FOOD101 DEMO</h1>
        <hr style='width: 60%; margin: auto; border: 1px solid #bbb; margin-bottom: 1.5em;'>
    """,
        unsafe_allow_html=True,
    )

    # General description
    st.markdown(
        f"""
        <p style='text-align:center; font-size: 1.1em; margin-bottom: 2.5em;'>
        Welcome to the <b>FOOD101 Demo App!</b><br>
        We have <b>{len(run_names)}</b> trained models for food classification.<br><br>
        Explore live predictions, metrics, and model comparison.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Pages description
    pages_info = [
        {
            "name": "📷 Live Prediction",
            "desc": "Upload your food images, choose a model, and see instant predictions with confidence scores.",
        },
        {
            "name": "📊 Metrics and Configuration",
            "desc": "Explore model accuracy, loss curves, and hyperparameters used during training.",
        },
        {
            "name": "📈 Compare All Models",
            "desc": "Analyze performance across all trained models and compare test results side by side.",
        },
    ]

    # Show sections and descriptions
    num_pages = len(pages_info)
    cols = st.columns(num_pages)

    for i, page_info in enumerate(pages_info):
        with cols[i]:
            st.markdown(
                f"""
                <div style='text-align:center; padding: 1.2em; border-radius: 15px; 
                            background-color: #f9f9f9; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                    <h3 style='margin-bottom: 0.4em;'>{page_info['name']}</h3>
                    <p style='font-size: 0.95em; color: #444;'>{page_info['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)


# ------------------------------------------
#             LIVE PREDICTIONS
# -----------------------------------------


def page_live_prediction(run_names):

    run_display_names = [f"Top {i+1}: {name}" for i, name in enumerate(run_names)]

    # --- Model selector ---
    selected_display = st.selectbox(
        "Choose a model for predictions",
        run_display_names,
        index=0,
        key="live_pred_model",
    )

    # Get original name
    selected_model = run_names[run_display_names.index(selected_display)]

    # set variables from the session_state
    model = st.session_state.loaded_models[selected_model]
    transform = st.session_state.model_transforms[selected_model]
    class_names = st.session_state.model_classnames[selected_model]
    model_name = selected_model
    model_info = st.session_state.model_info[selected_model]

    test_acc = model_info.get("test_metrics", {}).get("accuracy", 0) * 100

    col_left, col_right = st.columns([1, 1])

    # -------------------- Left column: uploader, predict button, results --------------------
    with col_left:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            label_visibility="visible",
        )

        # Reset if no file or new file
        if uploaded_file != st.session_state.get(
            "uploaded_file"
        ) or model_name != st.session_state.get("model_name"):
            st.session_state.uploaded_file = uploaded_file
            st.session_state.model_name = model_name
            st.session_state.uploaded_image = None
            st.session_state.prediction_result = None

        # Assign uploaded image
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")

        # Predict button
        predict_disabled = st.session_state.get("uploaded_image") is None
        if st.button(
            "🔮 Predict", type="primary", disabled=predict_disabled, width="stretch"
        ):
            result = predict_image(
                image_input=st.session_state.uploaded_image,
                model=model,
                transform=transform,
                class_names=class_names,
                device="cuda" if cuda.is_available() else "cpu",
            )
            st.session_state.prediction_result = result

        # Results below button
        if st.session_state.get("prediction_result") is not None:
            result = st.session_state.prediction_result
            render_prediction_card(
                result["pred_class"],
                result["pred_prob"],
                model_name=model_name,
                test_acc=test_acc,
            )
            render_probability_bars(
                class_names, result["probabilities"][0], threshold=0.1
            )

        # show classes in expandable menu
        with st.expander("See Classes"):
            n = len(class_names)
            mid = (n + 1) // 2
            c1, c2 = st.columns(2)
            for c in class_names[:mid]:
                c1.write(f"- {c}")
            for c in class_names[mid:]:
                c2.write(f"- {c}")

    # -------------------- Right column: preview uploaded image --------------------
    with col_right:
        if st.session_state.get("uploaded_image") is not None:
            st.image(st.session_state.uploaded_image, width=400)


# -----------------------------------
#         METRICS & CONFIG
# ------------------------------------


def page_metrics_and_config(run_map, run_names):

    run_display_names = [f"Top {i+1}: {name}" for i, name in enumerate(run_names)]

    # --- Model selector ---
    selected_display = st.selectbox(
        "Choose a model for predictions",
        run_display_names,
        index=0,
        key="metrics and config",
    )

    # Get original name
    selected_model = run_names[run_display_names.index(selected_display)]

    # set variables from the session_state
    class_names = st.session_state.model_classnames[selected_model]
    model_info = st.session_state.model_info[selected_model]
    hyperparameters = model_info["hyperparameters"]
    hyperparameters["epochs"] = model_info["best_epoch"] + 1

    # set run
    run = run_map[selected_model]

    # --- Metrics ---
    train_metrics = model_info.get("train_metrics", {})
    val_metrics = model_info.get("val_metrics", {})
    test_metrics = model_info.get("test_metrics", {})
    test_metrics_table = {
        k: v for k, v in test_metrics.items() if k != "confusion_matrix"
    }

    # Order the metrics
    all_metrics = sorted(
        set(train_metrics) | set(val_metrics) | set(test_metrics_table)
    )

    # Create dataframe with datasets as rows and metrics as columns
    df_metrics = pd.DataFrame(
        {
            "Dataset": ["Train", "Validation", "Test"],
        }
    )

    for metric in all_metrics:
        df_metrics[metric] = [
            train_metrics.get(metric, None),
            val_metrics.get(metric, None),
            test_metrics_table.get(metric, None),
        ]

    # Format floats
    df_metrics = df_metrics.applymap(
        lambda x: (
            f"{x:.4f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else x
        )
    )

    # --- Tabs ---
    tab_metrics, tab_confusion, tab_loss, tab_hparams = st.tabs(
        [
            "📊 Metrics Table",
            "🧩 Confusion Matrix",
            "📉 Loss Curve",
            "⚙️ Hyperparameters",
        ]
    )

    # --- Tab 1: Metrics ---
    with tab_metrics:
        st.subheader("📈 Model Metrics")
        st.dataframe(df_metrics, width="stretch")

    # --- Tab 2: Confusion Matrix ---
    with tab_confusion:
        st.subheader("🧩 Normalized Confusion Matrix (Test Set)")
        cm = model_info["test_metrics"].get("confusion_matrix", None)
        if cm is not None:
            render_confusion_matrix(np.array(cm), class_names)
        else:
            st.info("No confusion matrix found.")

    # --- Tab 3: Loss Curve ---
    with tab_loss:
        st.subheader("📉 Loss Curve")
        loss_plot_path = run["run_path"] / "artifacts" / "plots" / "loss_curve.png"
        if loss_plot_path.exists():
            st.image(loss_plot_path, width=700)
        else:
            st.info("No loss curve available.")

    # --- Tab 4: Hyperparameters ---
    with tab_hparams:
        st.subheader("⚙️ Model Hyperparameters")
        if hyperparameters:
            df_hparams = pd.DataFrame(
                [(k, str(v)) for k, v in hyperparameters.items()],
                columns=["Parameter", "Value"],
            )
            st.table(df_hparams)
        else:
            st.info("No hyperparameters found.")


# ----------------------------------
#         COMPARE MODELS
# -----------------------------------


def page_compare_models(runs):
    # --- Build df with all metrics---
    all_models_data = []
    for run in runs:
        info = run["info"]
        model_name = run["run_path"].name
        train = info.get("train_metrics", {})
        val = info.get("val_metrics", {})
        test = info.get("test_metrics", {})
        metrics_keys = set(train) | set(val) | set(test)
        for m in metrics_keys:
            all_models_data.append(
                {
                    "Model": model_name,
                    "Metric": m,
                    "Train": train.get(m, None),
                    "Validation": val.get(m, None),
                    "Test": test.get(m, None),
                }
            )

    if not all_models_data:
        st.info("No models found to compare.")
        return

    df = pd.DataFrame(all_models_data)

    # --- Crear pestañas por cada métrica ---
    all_metrics = df["Metric"].unique().tolist()
    all_metrics = [
        metric for metric in all_metrics if metric.lower() != "confusion_matrix"
    ]
    tabs = st.tabs(all_metrics)

    for metric, tab in zip(all_metrics, tabs):
        with tab:
            st.subheader(f"Metric: {metric}")

            df_metric = df[df["Metric"] == metric].copy()

            # --- Tabla ---
            table_display = df_metric[
                ["Model", "Train", "Validation", "Test"]
            ].set_index("Model")
            st.table(table_display)

            # --- Melt para formato largo ---
            df_melt = df_metric.melt(
                id_vars="Model",
                value_vars=["Train", "Validation", "Test"],
                var_name="Dataset",
                value_name="Value",
            )

            # --- Crear gráfico horizontal agrupado ---
            plt.figure(
                figsize=(8, 0.8 * len(df_metric))
            )  # altura dinámica según modelos

            sns.barplot(
                data=df_melt,
                x="Value",
                y="Model",
                hue="Dataset",
                dodge=True,  # grOUP BARS
            )

            plt.xlabel(metric)
            plt.yticks(fontsize=9)
            plt.legend(title="Dataset")
            plt.tight_layout()
            if metric != "loss":
                plt.xlim(0, 1.2)  # eje X va de 0 a 1.1

            st.pyplot(plt.gcf())  # mostrar gráfico en Streamlit
            plt.clf()  # limpiar figura para la siguiente pestaña

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def render_prediction_card(
    pred_class: str, pred_prob: float, model_name: str = None, test_acc: float = None
):
    """Show the prediction with optional model info below."""

    model_info_html = ""
    if model_name is not None:
        model_info_html += f'<p style="font-size:14px; color:#555; margin:5px 0 0 0;">Model Name: <b>{model_name}</b></p>'
    if test_acc is not None:
        model_info_html += f'<p style="font-size:14px; color:#555; margin:0;">Model Test Accuracy: <b>{test_acc:.2f}%</b></p>'

    st.markdown(
        f"""
        <div style="
            background-color:#e8f5e9; 
            border-radius:15px; 
            padding:20px; 
            text-align:center; 
            box-shadow:0px 0px 10px rgba(0,0,0,0.1); 
            margin-bottom:15px;
        ">
            <h3 style="color:#2e7d32;">✅ Predicted Class: <b>{pred_class}</b></h3>
            <p style="font-size:16px;">Confidence: <b>{pred_prob*100:.2f}%</b></p>
            {model_info_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(
    class_names: list, probabilities: list, threshold: float = 0.1
):
    """Show the probabilities grater than the threshold."""
    top_probs = [
        (cls, p) for cls, p in zip(class_names, probabilities) if p >= threshold
    ]
    top_probs.sort(key=lambda x: x[1], reverse=True)
    for cls_name, prob in top_probs:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; margin-bottom:6px;">
                <div style="flex:1; font-weight:bold;">{cls_name}</div>
                <div style="flex:3; background:#ddd; border-radius:8px; height:20px; margin:0 10px;">
                    <div style="width:{prob*100:.1f}%; background:#4CAF50; height:100%; border-radius:8px;"></div>
                </div>
                <div style="width:50px; text-align:right;">{prob*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_confusion_matrix(cm_percent: np.ndarray, class_names: list):
    """
    Displays a scalable confusion matrix using percentages with Altair.

    cm_percent: np.ndarray of shape (num_classes, num_classes), already in percentages
    class_names: list of class names (str)
    """
    num_classes = len(class_names)

    # --- Convert to long-format DataFrame ---
    df_cm = pd.DataFrame(cm_percent, index=class_names, columns=class_names)
    df_cm_long = df_cm.reset_index().melt(id_vars="index")
    df_cm_long.columns = ["True Class", "Predicted Class", "Percentage"]

    # --- Dynamic chart size ---
    cell_size = 40
    max_size = 800
    width = min(max_size, cell_size * num_classes)
    height = min(max_size, cell_size * num_classes)

    # --- Create interactive heatmap ---
    chart = (
        alt.Chart(df_cm_long)
        .mark_rect()
        .encode(
            x=alt.X("Predicted Class:O", sort=class_names),  # left → right
            y=alt.Y(
                "True Class:O", sort=class_names  # top → bottom, normal orientation
            ),
            color=alt.Color("Percentage:Q", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("True Class", title="True Class"),
                alt.Tooltip("Predicted Class", title="Predicted Class"),
                alt.Tooltip("Percentage:Q", title="Percentage", format=".1f"),
            ],
        )
        .properties(width=width, height=height)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

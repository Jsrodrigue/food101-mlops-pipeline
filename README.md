# food101

A project for training and evaluating food classification models using **PyTorch**, **MLflow**, and **Hydra**.

This project demonstrates a full **MLOps pipeline**: from automated data preparation, training, experiment tracking, and model selection, to testing and deployment of a live demo. All stages are designed to **maximize reproducibility and automation**, allowing you to run experiments, evaluate results, and deploy models with minimal manual intervention.

> **Note:** The pipeline is compatible with **any image classification dataset** that follows the same folder structure as `data/dataset/` (i.e., `train/`, `val/`, and `test/` folders, each containing one subfolder per class with images inside). You can replace the Food101 data with your own dataset as long as you keep this structure.

---


## 🖼️ App Screenshots

Below are some screenshots of the app in action:

![Home Page](images/screenshot_home.png)
![Live Prediction](images/screenshot_predict.png)
![Metrics Page](images/screenshot_metrics.png)

---

## 🎬 Video Demo

Watch a 10-minute walkthrough of the full pipeline and app usage:

[![Watch the demo](https://img.youtube.com/vi/TU_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=TU_VIDEO_ID_HERE)

Or click here: [Video Demo on YouTube](https://www.youtube.com/watch?v=TU_VIDEO_ID_HERE)

---

## 🚀 How to Use

You can use this project in **two ways**:

---

### 1. Run Only the Demo App

If you just want to try the app with already trained models (no need to run the full pipeline):

1. **Clone the repository and install dependencies:**

    ```bash
    git clone https://github.com/tu_usuario/food101Mini.git
    cd food101Mini
    conda create -n food101 python=3.10
    conda activate food101
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

2. **Make sure you have the dataset and trained models available locally.**

    The folders `data/dataset/` and `selected_models/` must exist on your machine.  
    If you have not run the pipeline, you need to prepare the dataset and train the models first.

3. **Launch the demo app:**

    ```bash
    make demo
    ```
    or
    ```bash
    streamlit run app.py
    ```

---

### 2. Full MLOps Pipeline (Reproducible Training & Evaluation)

Run the entire pipeline from data preparation to model selection and testing using the provided `makefile` commands.

#### **Step-by-step Pipeline**

| Command              | Description                          |
|----------------------|--------------------------------------|
| `make prepare`       | Prepare the dataset                  |
| `make experiments`   | Run all experiments                  |
| `make select`        | Select top models                    |
| `make test`          | Test selected models                 |
| `make demo`          | Launch Streamlit demo (if available) |
| `make clean`         | Remove outputs and logs              |
| `make ui`            | Launch MLflow UI                     |
| `make help`          | Show all available commands          |

**Example usage:**

```bash
make prepare
make experiments
make select
make test
make demo
```

> This makes it much easier to run the full pipeline without typing long commands.

#### **Configuration-driven Workflow**

- **Experiments:**  
  Defined in `conf/experiments.yaml` (models, hyperparameters, augmentations, etc.).
- **Model Selection:**  
  Controlled by `conf/select_models.yaml` (top_k, metric, etc.).
- **Testing:**  
  Controlled by `conf/test.yaml` (batch size, device, metrics, etc.).

All results and artifacts are tracked with MLflow (`mlruns/`).

---

### ⚠️ Important Note for Full Pipeline

If you want to run the **full pipeline** (starting from data preparation with `make prepare`), make sure to **delete the `selected_models/` folder** after cloning the repository.  
This is important because if `selected_models/` exists from a previous run (possibly with different classes), it may cause errors or inconsistencies when preparing a new dataset with different classes.

You can safely remove it with:

```bash
rm -rf selected_models
```
or on Windows:
```cmd
rmdir /s /q selected_models
```

Then proceed with the pipeline as usual:

```bash
make prepare
make experiments
make select
make test
make demo
```

---

### ⚡️ Orchestrate the Full Pipeline with a Single Command

You can also run the **entire pipeline automatically** using the provided `orchestrator` script.  
This script will execute all the main steps (prepare, experiments, select, test, etc.) in the correct order.

Simply run:

```bash
make run
```
or
```bash
python -m scripts.orchestrator
```

This is especially useful for reproducibility, automation, or when running the pipeline on a new dataset from scratch.

---

## 📂 Project Structure

Below is the recommended folder structure for this project.  
**Note:** The `data/dataset/` folder (and its subfolders) will not appear until you prepare or generate your dataset locally, since data is not included in the repository.

```
food101/
├── conf/                # Hydra configuration files
│   ├── experiments.yaml
│   ├── select_models.yaml
│   ├── test.yaml
│   └── ... 
├── data/                # Data folder (not versioned, created locally)
│   └── dataset/
│       ├── train/
│       ├── val/
│       └── test/
├── mlruns/              # MLflow logs and artifacts (not versioned)
├── outputs/             # Experiment outputs (not versioned)
├── selected_models/     # Selected/best models (not versioned)
├── scripts/             # Pipeline and utility scripts
│   ├── save_data.py
│   ├── run_experiments.py
│   ├── select_models.py
│   ├── train.py
│   ├── test.py
│   └── orchestrator.py
├── src/                 # Source code
│   ├── models/
│   ├── utils/
│   ├── predictions.py
│   └── st_sections.py
├── app.py               # Streamlit or Gradio app
├── requirements.txt
├── makefile
├── README.md
└── .gitignore
```

> **Note:**  
> Folders like `data/dataset/`, `mlruns/`, `outputs/`, and `selected_models/` are **not included** in the repository and will be created automatically as you run the pipeline.

---

## ⚙️ Configuration Files

Each main step of the pipeline uses YAML configuration files in the `conf/` folder:

- **Experiments (`make experiments`):**  
  Uses `conf/experiments.yaml` to define models, hyperparameters, augmentations, number of epochs, etc. The script automatically launches all experiments and logs results to MLflow.

- **Select Top Models (`make select`):**  
  Uses `conf/select_models.yaml` to define `top_k`, metric name, MLflow runs directory, and target folder for copying the best models.

- **Test Selected Models (`make test`):**  
  Reads configuration from `conf/test.yaml` for batch size, device (`cpu` or `cuda`), metrics, and saving options.

This approach separates code logic from experimental parameters, making workflows reproducible, easier to compare, and collaborative.

---

## 🧹 Cleaning Outputs

To remove all outputs, selected models, and MLflow logs:

```bash
make clean
```

This safely deletes `outputs/`, `selected_models/`, and `mlruns/` if they exist.

---

## 💡 Additional Recommendations

- Launch MLflow UI with `make ui` for experiment tracking and exploring all the runs.
- Run `make help` for all available commands.
- Add a video demo or screenshots to further enhance your portfolio!

---


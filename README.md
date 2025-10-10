# food101Mini

A project for training and evaluating food classification models using **PyTorch**, **MLflow**, and **Hydra**.  

This project demonstrates a full **MLOps pipeline**: from automated data preparation, training, experiment tracking, and model selection, to testing and deployment of a live demo. All stages are designed to **maximize reproducibility and automation**, allowing you to run experiments, evaluate results, and deploy models with minimal manual intervention.  

By leveraging Hydra for configuration management, MLflow for experiment tracking, and modular scripts, this project allows you to:  

- Run multiple experiments with different models, hyperparameters, and augmentations automatically.  
- Track and compare model performance metrics without manually editing scripts.  
- Select top-performing models based on any evaluation metric in an automated way.  
- Test selected models and generate structured evaluation reports.  
- Deploy a demo app for inference on new images, demonstrating the complete workflow.


---

## How to Run the Project

You can use the provided `makefile` to simplify running the main steps of the project.  
Here are the available commands:

## Quick Overview of Commands

| Command        | What it does (summary)                                                |
|----------------|---------------------------------------------                          |
| `make prepare` | Prepare dataset: download, extract, split into train/val/test.        |
| `make experiments` | Run all configured experiments and log results to MLflow.         |
| `make select`  | Pick top-K models based on a metric like accuracy or F1.              |
| `make test`    | Evaluate the selected models on the test set and save metrics.        |
| `make demo`    | Launch Streamlit demo (optional, if installed).                       |
| `make clean`   | Remove outputs, selected models, and MLflow logs.                     |
| `make ui`      | Open MLflow UI to explore experiments.                                |
| `make help`    | Show all available commands.                                          |
| `make run`     | Run the full pipeline: prepare → experiments → select → test → demo.  | 

**Example usage:**

```bash
make prepare
make experiments
make select
make test
make demo
```

> This makes it much easier to run the full pipeline without typing long commands.

---

### 1. Clone the Repository

```bash
git clone https://github.com/tu_usuario/food101Mini.git
cd food101Mini
```

---

### 2. Install Dependencies

It is recommended to use a virtual environment:

#### On Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### On macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Data Preparation

You can create a custom subset of Food101 using the `create_data` function in `data_engine.py`.  

This function allows you to:

- **Select classes manually** by specifying a list in the config (`selected_classes`).  
- **Select a random subset of classes** by specifying the number of classes and using `select_mode: "random"`.  
- **Select the first N classes alphabetically** with `select_mode: "first"` and `num_classes`.  

The script will automatically:

- Download and verify the Food101 dataset if not present or if the zip is corrupt.  
- Create balanced train/val/test splits for the selected classes.  
- Save a class mapping file (`class_map.json`) for reproducibility.  

> **Note:** If you only want to use the included dataset under `data/dataset`, you do **not** need to run this step.

---

### 4. Run Experiments

To train and evaluate your models as defined in your configuration, run:

```bash
make experiments
```

This will execute all experiments specified in your configuration files and log results to MLflow.

---

### 5. Select Top Models

After running experiments, select the top-performing models based on your chosen metric:

```bash
make select
```

- Configure selection in `conf/select_models.yaml`.

Example configuration:

```yaml
top_k: 3
metric_name: "accuracy"  # Options: "accuracy", "f1_macro", etc.
source_runs_dir: mlruns
target_selected_models_dir: selected_models
```

---

### 6. Test Selected Models

To evaluate the selected top models on the test set, run:

```bash
make test
```

- Configure testing in `conf/test.yaml`.

Example configuration:

```yaml
runs_dir: "selected_models"
batch_size: 32
save_results: true
device: cpu
loss_fn: CrossEntropyLoss
metrics: ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
```

---

### 7. Project Structure

```
food101Mini/
├── conf/                # Hydra configuration files
├── data/                # Training, validation, and test datasets
│   └── dataset/
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/             # Execution scripts
│   ├── save_data.py
│   ├── run_experiments.py
│   ├── select_models.py
│   ├── train.py
│   └── test.py
├── src/                 # Project source code
│   ├── models/          # Model definitions
│   └── utils/           # Utility functions
├── outputs/             # Experiment outputs
├── selected_models/     # Top-K selected models
└── mlruns/              # MLflow logs
```

---

### 8. How Configuration Files Drive the Pipeline

Each main step of the pipeline uses YAML configuration files in the `conf/` folder.  

- **Experiments (`make experiments`):**  
  Uses `conf/experiments.yaml` to define models, hyperparameters, augmentations, number of epochs, etc. The script automatically launches all experiments and logs results to MLflow.

- **Select Top Models (`make select`):**  
  Uses `conf/select_models.yaml` to define `top_k`, metric name, MLflow runs directory, and target folder for copying the best models.

- **Test Selected Models (`make test`):**  
  Reads configuration from `conf/test.yaml` for batch size, device (`cpu` or `cuda`), metrics, and saving options.

**Why is this useful?**  
It separates code logic from experimental parameters, making workflows reproducible, easier to compare, and collaborative.

---

### 9. Cleaning Outputs

To remove all outputs, selected models, and MLflow logs:

```bash
make clean
```

This safely deletes `outputs/`, `selected_models/`, and `mlruns/` if they exist.

---

### 10. Additional Recommendations

- Optional: Launch MLflow UI with `make ui` for experiment tracking and exploring all the runs.  
- Deploy or test the demo app with `make demo`.  
- Run `make help` for all available commands.

---
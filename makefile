# -----------------------
# Launch MLflow UI
# -----------------------
ui:
	cd C:/Users/Juan/Desktop/food101Mini
	mlflow ui --backend-store-uri mlruns



# -----------------------
# Prepare the dataset
# -----------------------
prepare:
	python -m scripts.save_data

# -----------------------
# Train with the default config
# -----------------------
train:
	python -m scripts.train

# -----------------------
# Run experiments
# -----------------------
experiments:
	python -m scripts.run_experiments

# -----------------------
# Select top models
# -----------------------
select:
	python -m scripts.select_models

# -----------------------
# Test selected models
# -----------------------
test:
	python -m scripts.test

# -----------------------
# Launch Streamlit demo
# -----------------------
demo:
	streamlit run app.py

# -----------------------
# Clean outputs and logs
# -----------------------
clean:
	rm -rf outputs selected_models mlruns || true

#------------------------
# Run full pipeline
#------------------------
run:
	@echo "Running full ML pipeline..."
	python -m scripts.orchestrator

# -----------------------
# Help
# -----------------------
help:
	@echo "Available commands:"
	@echo "  make ui         # Launch MLflow UI"
	@echo "  make prepare    # Prepare the dataset"
	@echo "  make experiments# Run all experiments"
	@echo "  make select     # Select top models"
	@echo "  make test       # Test selected models"
	@echo "  make train      # Train the model"
	@echo "  make demo       # Launch Streamlit demo"
	@echo "  make clean      # Remove outputs and logs"


# orchestrator.py
import subprocess
import time

if __name__ == "__main__":

    modules = [
        "scripts.save_data",
        "scripts.run_experiments",
        "scripts.select_models",
        "scripts.test",
    ]

    for module in modules:
        print(f"\n🚀 Running: {module}")
        start = time.time()
        subprocess.run(["python", "-m", module], check=True)
        print(f"✅ Finished {module} in {time.time() - start:.1f}s")

    # Launch Streamlit app
    subprocess.run(["streamlit", "run", "app.py"])

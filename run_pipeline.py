import os
from core.config import CONFIG
from core.dataset_loader import get_data_loaders
from train import run_research_training
from evaluate_comparison import evaluate_model_performance

def start_pipeline():
    print(" --- STARTING RESEARCH PIPELINE --- ")
    
    
    
    for folder in [CONFIG["model_dir"], CONFIG["results_dir"]]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f" Created folder: {folder}")

    
    print("\n Phase 1: Training with Math-Driven Loss...")
    run_research_training()

    # ۳. فاز ارزیابی (Evaluation)
    print("\n Phase 2: Scientific Evaluation & Benchmarking...")
    evaluate_model_performance()

    print("\n --- PIPELINE COMPLETED SUCCESSFULLY --- ")
    print(f" Model and Results are ready for Deployment.")

if __name__ == "__main__":
    start_pipeline()
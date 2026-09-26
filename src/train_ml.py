"""
train_ml.py
-----------
Classical Machine Learning Training Pipeline for Student Engagement Detection.
Wrapper / entrypoint matching the prompt command:
    python src/train_ml.py
"""
import sys
from pathlib import Path

# Delegate to train.py main
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import main

if __name__ == "__main__":
    main()

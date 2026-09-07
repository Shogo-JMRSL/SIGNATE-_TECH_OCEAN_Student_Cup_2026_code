from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.competition_pipeline import run_pipeline


if __name__ == "__main__":
    result = run_pipeline(PROJECT_DIR)
    metrics = result["metrics"]
    print("Selected model:", metrics["selected_model"])
    print("OOF F1 (tuned threshold):", f'{metrics["oof_f1_tuned"]:.4f}')
    print("Cross-fitted threshold F1:", f'{metrics["crossfit_threshold_f1"]:.4f}')
    print("Tail stress-test F1:", f'{metrics["tail_stress_test"]["holdout_f1"]:.4f}')
    print("Submission positives:", metrics["predicted_positive_count"])

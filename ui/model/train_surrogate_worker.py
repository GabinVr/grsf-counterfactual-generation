"""
Worker script that runs surrogate model training in an isolated Python process.

Invoked by SurrogateModelObject.train() via subprocess to avoid Streamlit's
ScriptRunner interfering with PyTorch training.

Usage:
    python train_surrogate_worker.py <input_pickle> <output_pickle>

Input pickle format (tuple):
    (X_train, y_train, X_test, y_test, grsf_model, model_class_name,
     model_params, epochs, learning_rate)

Output pickle format (dict):
    Success: {'model': trained_model, 'accuracy': float,
              'approximation_metrics': {'agreement': float, 'surrogate_accuracy': float},
              'model_arch': str}
    Failure: {'error': traceback_string}
"""
import os
import pickle
import sys
import traceback

UI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(UI_ROOT)
if UI_ROOT not in sys.path:
    sys.path.insert(0, UI_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main(input_path: str, output_path: str) -> int:
    try:
        import models
        import torch

        with open(input_path, "rb") as f:
            (X_train, y_train, X_test, y_test, grsf_model,
             model_class_name, model_params, epochs, learning_rate) = pickle.load(f)

        model_class = getattr(models, model_class_name)
        model = model_class(**model_params)

        model.train(X_train, y_train, epochs=epochs, lr=learning_rate, debug=True)

        accuracy = model.evaluate(X_test, y_test)

        # Approximation metrics: agreement between surrogate and GRSF predictions
        surrogate_predictions = model.predict(X_test)
        if isinstance(surrogate_predictions, torch.Tensor):
            surrogate_predictions = surrogate_predictions.cpu().numpy()
        grsf_predictions = grsf_model.predict(X_test)
        agreement = (surrogate_predictions == grsf_predictions).mean() * 100

        result = {
            "model": model,
            "accuracy": accuracy,
            "approximation_metrics": {
                "agreement": agreement,
                "surrogate_accuracy": accuracy,
            },
            "model_arch": model.get_architecture(),
        }
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        return 0
    except Exception:
        with open(output_path, "wb") as f:
            pickle.dump({"error": traceback.format_exc()}, f)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: train_surrogate_worker.py <input_pickle> <output_pickle>",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

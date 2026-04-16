"""
Worker script that runs counterfactual generation in an isolated Python process.

Invoked by LocalCounterfactualGeneratorObject.generate() and
GlobalCounterfactualGeneratorObject.generate() via subprocess to avoid
Streamlit's ScriptRunner interfering with PyTorch gradient-based optimization.

Usage:
    python generate_counterfactual_worker.py <input_pickle> <output_pickle>

Input pickle format (dict), with a 'mode' key of either 'local' or 'batch'.

To keep the cross-process payload small and safe, the surrogate is shipped as
(class_name, params, state_dict) and reconstructed here; numpy arrays are used
for tensor inputs and converted back to torch inside the worker.

    Local mode:
        {
            'mode': 'local',
            'grsf_model': ...,                # sklearn; pickles fine
            'surrogate_class_name': str,      # attribute on `models`
            'surrogate_params': dict,         # kwargs for the class
            'surrogate_state_dict': dict,     # CPU tensors
            'base_sample': np.ndarray,
            'base_class': int,
            'target_sample': np.ndarray,
            'binary_mask': np.ndarray | None,
            'parameters': {'epochs': int, 'learning_rate': float, 'beta': float},
        }

    Batch mode:
        {
            'mode': 'batch',
            'grsf_model': ...,
            'surrogate_class_name': str,
            'surrogate_params': dict,
            'surrogate_state_dict': dict,
            'split_dataset': (X_train, y_train, X_test, y_test),  # numpy
            'nb_samples': int,
            'parameters': {'epochs': int, 'learning_rate': float, 'beta': float},
        }

Output pickle format (dict):
    Success (local):
        {'counterfactual': np.ndarray, 'training_progress': str}
    Success (batch):
        {'counterfactuals': list, 'training_progress': list}
    Failure:
        {'error': traceback_string}
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


def _build_surrogate(class_name: str, params: dict, state_dict: dict):
    """Rebuild a surrogate nn.Module from class name, params, and state_dict.

    The surrogate is frozen (eval mode, requires_grad=False on all parameters):
    counterfactual optimization only propagates gradients through the input
    sample ``x``, not through the model weights.

    Note: ``BaseSurrogateClassifier`` overrides ``nn.Module.train`` to be a
    fitting method, so ``model.eval()`` (which calls ``self.train(False)``)
    cannot be used. Flip the ``training`` flag manually on the top-level
    module and let ``nn.Module.train`` do the recursive work on children.
    """
    import models

    model_cls = getattr(models, class_name)
    model = model_cls(**params)
    model.load_state_dict(state_dict)
    model.training = False
    for child in model.children():
        child.train(False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_local(payload: dict) -> dict:
    import torch
    from gen import CounterFactualCrafting

    training_progress = ""

    def training_callback(epoch: int, loss: float) -> str:
        nonlocal training_progress
        line = f"Epoch {epoch + 1}: Loss = {loss:.4f}\n"
        training_progress += line
        return line

    surrogate = _build_surrogate(
        payload["surrogate_class_name"],
        payload["surrogate_params"],
        payload["surrogate_state_dict"],
    )

    base_sample = torch.tensor(payload["base_sample"], dtype=torch.float32)
    target_sample = torch.tensor(payload["target_sample"], dtype=torch.float32)
    binary_mask = payload["binary_mask"]
    if binary_mask is not None:
        binary_mask = torch.tensor(binary_mask, dtype=torch.float32)

    crafter = CounterFactualCrafting(
        payload["grsf_model"],
        surrogate,
    )

    params = payload["parameters"]
    counterfactual = crafter.generate_local_counterfactuals(
        target_sample,
        base_sample,
        payload["base_class"],
        binary_mask,
        lr=params["learning_rate"],
        epochs=params["epochs"],
        beta=params["beta"],
        debug=True,
        training_callback=training_callback,
    )

    if torch.is_tensor(counterfactual):
        counterfactual = counterfactual.detach().cpu().numpy()

    return {
        "counterfactual": counterfactual,
        "training_progress": training_progress,
    }


def run_batch(payload: dict) -> dict:
    from counterfactual import counterfactual_batch_generation

    training_progress = []

    def training_callback(data: str) -> str:
        training_progress.append(data)
        return data

    surrogate = _build_surrogate(
        payload["surrogate_class_name"],
        payload["surrogate_params"],
        payload["surrogate_state_dict"],
    )

    params = payload["parameters"]
    counterfactuals = counterfactual_batch_generation(
        grsf_classifier=payload["grsf_model"],
        nn_classifier=surrogate,
        split_dataset=payload["split_dataset"],
        nb_samples=payload["nb_samples"],
        epochs=params["epochs"],
        lr=params["learning_rate"],
        beta=params["beta"],
        debug=True,
        training_callback=training_callback,
    )

    return {
        "counterfactuals": counterfactuals,
        "training_progress": training_progress,
    }


def main(input_path: str, output_path: str) -> int:
    try:
        with open(input_path, "rb") as f:
            payload = pickle.load(f)

        mode = payload.get("mode")
        if mode == "local":
            result = run_local(payload)
        elif mode == "batch":
            result = run_batch(payload)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        return 0
    except Exception:
        with open(output_path, "wb") as f:
            pickle.dump({"error": traceback.format_exc()}, f)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_counterfactual_worker.py <input_pickle> <output_pickle>",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

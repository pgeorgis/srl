"""Locate a trained SRL model."""
import os
from pathlib import Path

from .constants import DEFAULT_OUTDIR_ROOT, MODEL_DIR_ENV_VAR


def find_model(model_path: str | os.PathLike = None) -> str:
    """Return the location of a trained SRL model to load.

    Search order: `model_path` (a local directory or a Hugging Face Hub model ID),
    then the directory in the environment variable SRL_MODEL_DIR.
    """
    if model_path is not None:
        return str(model_path)
    env_dir = os.environ.get(MODEL_DIR_ENV_VAR)
    if env_dir:
        directory = Path(env_dir).expanduser()
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(
                f"${MODEL_DIR_ENV_VAR} is set to '{env_dir}', but no model (config.json) was found there."
            )
        return str(directory)
    raise FileNotFoundError(
        "No trained SRL model specified. Pass `model_path` (a local directory or Hugging Face model ID) "
        f"or set the environment variable {MODEL_DIR_ENV_VAR} to the directory of a trained model."
    )


def default_model_outdir(base_model: str | os.PathLike, root: str | os.PathLike = DEFAULT_OUTDIR_ROOT) -> str:
    """Return the default output directory for a model fine-tuned from `base_model`: <root>/<base model>.

    A Hugging Face Hub model ID keeps its namespace as a sub-directory
    (e.g. "model-family/my-specific-model" -> "out/model-family/my-specific-model");
    a local base model directory contributes only its own name (e.g. "/home/Documents/models/my-xlmr/" -> "out/my-xlmr").
    """
    base_model_path = Path(base_model).expanduser()
    name = base_model_path.resolve().name if base_model_path.exists() else str(base_model).strip("/")
    return str(Path(root) / name)

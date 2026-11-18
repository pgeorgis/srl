from pathlib import Path

import pytest

from srl import find_model
from srl.constants import DEFAULT_OUTDIR_ROOT, MODEL_DIR_ENV_VAR
from srl.paths import default_model_outdir


def test_find_model_explicit_path_is_returned_as_is(monkeypatch):
    monkeypatch.delenv(MODEL_DIR_ENV_VAR, raising=False)
    assert find_model("some/dir") == "some/dir"


def test_find_model_from_env(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(tmp_path))
    assert find_model() == str(tmp_path)


def test_find_model_env_without_model_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        find_model()


def test_find_model_unspecified_raises(monkeypatch):
    monkeypatch.delenv(MODEL_DIR_ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError):
        find_model()


@pytest.mark.parametrize("base_model, expected", [
    pytest.param("liaad/srl-en_xlmr-base", "liaad/srl-en_xlmr-base", id="hub_id_with_namespace"),
    pytest.param("xlm-roberta-base", "xlm-roberta-base", id="hub_id_without_namespace"),
])
def test_default_model_outdir_from_hub_id(base_model, expected):
    assert default_model_outdir(base_model) == str(Path(DEFAULT_OUTDIR_ROOT) / expected)
    assert default_model_outdir(base_model, root="elsewhere") == str(Path("elsewhere") / expected)


def test_default_model_outdir_from_local_base_model(tmp_path):
    base_model = tmp_path / "models" / "my-xlmr"
    base_model.mkdir(parents=True)
    assert default_model_outdir(base_model) == str(Path(DEFAULT_OUTDIR_ROOT) / "my-xlmr")
    assert default_model_outdir(f"{base_model}/") == str(Path(DEFAULT_OUTDIR_ROOT) / "my-xlmr")

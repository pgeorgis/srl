
import pytest

from srl import find_model
from srl.constants import MODEL_DIR_ENV_VAR


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

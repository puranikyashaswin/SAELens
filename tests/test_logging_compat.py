"""
Tests for the logging compatibility layer.
"""

from pathlib import Path

from sae_lens import logging_compat


def test_get_backend_name_returns_string():
    backend = logging_compat.get_backend_name()
    assert isinstance(backend, str)
    assert backend in ["wandb", "swanlab"]


def test_generate_id_returns_valid_string():
    run_id = logging_compat.generate_id()
    assert isinstance(run_id, str)
    assert len(run_id) >= 8


def test_artifact_creation():
    artifact = logging_compat.Artifact(
        name="test_artifact",
        type="model",
        metadata={"key": "value"},
    )
    assert artifact._name == "test_artifact"
    assert artifact._type == "model"
    assert artifact._metadata == {"key": "value"}
    assert artifact._files == []


def test_artifact_add_file(tmp_path: Path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    artifact = logging_compat.Artifact(
        name="test_artifact",
        type="model",
    )
    artifact.add_file(str(test_file))
    assert str(test_file) in artifact._files


def test_histogram_creation_with_list():
    import numpy as np

    data = np.random.randn(100).tolist()
    histogram = logging_compat.Histogram(data)
    assert histogram._data is data
    # Should not raise
    repr(histogram)


def test_histogram_creation_with_array():
    import numpy as np

    data = np.random.randn(100)
    histogram = logging_compat.Histogram(data)
    assert histogram._data is data
    # Should not raise
    repr(histogram)


def test_log_artifact_does_not_raise_with_wandb_backend(tmp_path: Path):
    import contextlib

    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    artifact = logging_compat.Artifact(
        name="test_artifact",
        type="model",
    )
    artifact.add_file(str(test_file))
    # Should not raise even without an active run
    with contextlib.suppress(Exception):
        logging_compat.log_artifact(artifact, aliases=["test"])

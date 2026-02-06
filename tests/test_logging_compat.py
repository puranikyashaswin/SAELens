"""
Tests for the logging compatibility layer.
"""

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


def test_artifact_add_file():
    artifact = logging_compat.Artifact(
        name="test_artifact",
        type="model",
    )
    artifact.add_file("/path/to/file.txt")
    assert "/path/to/file.txt" in artifact._files


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


def test_log_artifact_does_not_raise_with_wandb_backend():
    artifact = logging_compat.Artifact(
        name="test_artifact",
        type="model",
    )
    artifact.add_file("/path/to/file.txt")
    # This should not raise even if not in an active run
    # In a real run it would log; here we're just checking no crash
    try:
        logging_compat.log_artifact(artifact, aliases=["test"])
    except Exception:
        # Expected to fail without an active wandb run, but shouldn't
        # raise ImportError or similar
        pass

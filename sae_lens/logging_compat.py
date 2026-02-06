"""
Logging backend compatibility layer for SAELens.

Supports wandb (default) and swanlab backends.
Set SAE_LENS_LOGGING_BACKEND=swanlab environment variable to use SwanLab.

SwanLab is an alternative experiment tracking tool that can serve as a drop-in
replacement for wandb in environments with network restrictions.

Note: SwanLab does not support Artifacts or Histograms. When using SwanLab:
- log_artifact() calls are silently skipped
- Histogram objects are represented as simple data summaries
"""

import os
import uuid
from typing import Any

_BACKEND = os.environ.get("SAE_LENS_LOGGING_BACKEND", "wandb").lower()

# Lazy backend loading to avoid import errors when optional dependency not installed
_backend_module: Any = None


def _get_backend() -> Any:
    """Lazily load the logging backend module."""
    global _backend_module
    if _backend_module is None:
        if _BACKEND == "swanlab":
            try:
                import swanlab  # type: ignore[import-not-found]

                _backend_module = swanlab
            except ImportError as e:
                raise ImportError(
                    "SwanLab is not installed. Install it with: pip install swanlab"
                ) from e
        else:
            import wandb

            _backend_module = wandb
    return _backend_module


def get_backend_name() -> str:
    """Return the name of the current logging backend."""
    return _BACKEND


def generate_id() -> str:
    """
    Generate a unique run ID.

    Uses wandb.util.generate_id() when available, falls back to UUID.
    """
    backend = _get_backend()
    if hasattr(backend, "util") and hasattr(backend.util, "generate_id"):
        return backend.util.generate_id()  # type: ignore
    return uuid.uuid4().hex[:8]


def init(
    project: str | None = None,
    entity: str | None = None,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    id: str | None = None,  # noqa: A002
    **kwargs: Any,
) -> Any:
    """
    Initialize a logging run.

    Args:
        project: Project name
        entity: Entity/team name
        config: Configuration dictionary to log
        name: Run name
        id: Run ID
        **kwargs: Additional backend-specific arguments
    """
    backend = _get_backend()
    return backend.init(
        project=project,
        entity=entity,
        config=config,
        name=name,
        id=id,
        **kwargs,
    )


def log(data: dict[str, Any], step: int | None = None) -> None:
    """
    Log metrics/data to the current run.

    Args:
        data: Dictionary of metrics to log
        step: Optional step number
    """
    backend = _get_backend()
    if step is not None:
        backend.log(data, step=step)
    else:
        backend.log(data)


def finish() -> None:
    """Finish the current logging run."""
    backend = _get_backend()
    backend.finish()


class Artifact:
    """
    Wrapper for logging artifacts (model files, etc).

    Note: SwanLab does not support artifacts. When using SwanLab backend,
    artifacts are tracked locally but not uploaded.
    """

    def __init__(
        self,
        name: str,
        type: str,  # noqa: A002
        metadata: dict[str, Any] | None = None,
    ):
        self._name = name
        self._type = type
        self._metadata = metadata
        self._files: list[str] = []

        if _BACKEND == "wandb":
            backend = _get_backend()
            self._artifact = backend.Artifact(name, type=type, metadata=metadata)
        else:
            self._artifact = None

    def add_file(self, path: str) -> None:
        """Add a file to the artifact."""
        if self._artifact is not None:
            self._artifact.add_file(path)
        self._files.append(path)


def log_artifact(artifact: Artifact, aliases: list[str] | None = None) -> None:
    """
    Log an artifact to the current run.

    Note: When using SwanLab backend, this is a no-op since SwanLab
    does not support artifact logging.

    Args:
        artifact: The artifact to log
        aliases: Optional list of aliases for the artifact
    """
    if _BACKEND == "wandb" and artifact._artifact is not None:
        backend = _get_backend()
        backend.log_artifact(artifact._artifact, aliases=aliases)
    # SwanLab: silently skip artifact logging


class Histogram:
    """
    Wrapper for histogram data visualization.

    Note: SwanLab does not support histograms. When using SwanLab backend,
    this class provides a simple representation of the data.
    """

    def __init__(self, data: Any):
        self._data = data

        if _BACKEND == "wandb":
            backend = _get_backend()
            self._histogram = backend.Histogram(data)
        else:
            self._histogram = None

    def __repr__(self) -> str:
        if self._histogram is not None:
            return repr(self._histogram)
        # For SwanLab, provide a simple summary
        if hasattr(self._data, "shape"):
            return f"Histogram(shape={self._data.shape})"
        if hasattr(self._data, "__len__"):
            return f"Histogram(len={len(self._data)})"
        return "Histogram(data)"

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class CADCompilationResult:
    """Outcome of compiling and rendering a generated CAD program."""

    success: bool
    stl_path: Optional[Path]
    image_paths: list[Path]
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None

    @property
    def rendering_paths(self) -> list[str]:
        """Return image paths as strings for downstream consumers."""
        return [str(path) for path in self.image_paths]

    def as_dict(self) -> dict:
        """Return a JSON-serialisable payload for persistence."""
        return {
            "success": self.success,
            "stl_path": str(self.stl_path) if self.stl_path else None,
            "image_paths": [str(path) for path in self.image_paths],
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
        }


class CADExecutor:
    """
    Wraps the sandboxed execution of generated CAD code.

    The executor writes the code to a temporary directory and invokes
    `evaluate/runtime.py` in a subprocess. This isolates user code from the
    orchestrator while producing the STL artefact and multi-view renderings.
    """

    def __init__(
        self,
        *,
        runtime_path: Optional[Path] = None,
        views: Optional[Iterable[str]] = None,
        python_executable: Optional[str] = None,
        timeout_seconds: int = 120,
    ) -> None:
        self._runtime_path = runtime_path or Path(__file__).with_name("runtime.py")
        self._views = list(views or ("front", "side", "top", "iso"))
        self._python = python_executable or sys.executable
        self._timeout = timeout_seconds
        self._logger = logging.getLogger(__name__)

    def compile(
        self,
        *,
        cad_code: str,
        artefact_dir: Path,
        sample_id: str,
    ) -> CADCompilationResult:
        artefact_dir.mkdir(parents=True, exist_ok=True)
        stl_output = artefact_dir / f"{sample_id}.stl"
        image_dir = artefact_dir / f"{sample_id}_renders"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Save the generated CAD code to persistent storage
        cad_script_persistent = artefact_dir / "generated_cad.py"
        cad_script_persistent.write_text(cad_code)

        with tempfile.TemporaryDirectory(prefix="cad_code_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            cad_script = tmp_path / "generated_cad.py"
            cad_script.write_text(cad_code)

            cmd = [
                self._python,
                str(self._runtime_path),
                "--cad-script",
                str(cad_script),
                "--stl-output",
                str(stl_output),
                "--image-dir",
                str(image_dir),
                "--views",
                ",".join(self._views),
            ]

            # Disable tokenizer parallelism in the subprocess to avoid warnings/deadlocks
            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"

            self._logger.info(
                "Sample %s: launching runtime with command %s",
                sample_id,
                " ".join(cmd),
            )
            start = time.perf_counter()
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:  # pragma: no cover
                elapsed = time.perf_counter() - start
                self._logger.error(
                    "Sample %s: runtime timed out after %.2fs. stdout=%s stderr=%s",
                    sample_id,
                    elapsed,
                    exc.stdout,
                    exc.stderr,
                )
                return CADCompilationResult(
                    success=False,
                    stl_path=None,
                    image_paths=[],
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    error=f"Execution timed out after {self._timeout}s",
                )

        stdout = completed.stdout
        stderr = completed.stderr
        elapsed = time.perf_counter() - start
        self._logger.info(
            "Sample %s: runtime finished in %.2fs with exit code %d",
            sample_id,
            elapsed,
            completed.returncode,
        )

        if completed.returncode != 0:
            self._logger.error(
                "Sample %s: runtime failed. stdout=%s stderr=%s",
                sample_id,
                stdout,
                stderr,
            )
            return CADCompilationResult(
                success=False,
                stl_path=None,
                image_paths=[],
                stdout=stdout,
                stderr=stderr,
                error=f"Runtime exited with status {completed.returncode}",
            )

        try:
            payload = json.loads(stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            self._logger.error(
                "Sample %s: runtime output was not valid JSON. stdout=%s stderr=%s",
                sample_id,
                stdout,
                stderr,
            )
            return CADCompilationResult(
                success=False,
                stl_path=None,
                image_paths=[],
                stdout=stdout,
                stderr=stderr,
                error=f"Failed to parse runtime JSON: {exc}",
            )

        if not payload.get("success", False):
            self._logger.error(
                "Sample %s: runtime reported failure payload=%s",
                sample_id,
                payload,
            )
            return CADCompilationResult(
                success=False,
                stl_path=None,
                image_paths=[],
                stdout=stdout,
                stderr=stderr,
                error=str(payload.get("error", "Unknown runtime failure")),
            )

        image_paths = [
            Path(path) for path in payload.get("image_paths", []) if path
        ]
        return CADCompilationResult(
            success=True,
            stl_path=Path(payload["stl_path"]),
            image_paths=image_paths,
            stdout=stdout,
            stderr=stderr,
            error=None,
        )

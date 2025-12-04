from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, List


def load_generated_module(script_path: Path):
    """
    Dynamically import the generated CAD module from disk.

    The caller is responsible for ensuring the file exists and contains valid Python code.
    """
    spec = importlib.util.spec_from_file_location("generated_cad_module", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to load CAD module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Some generated scripts rely on CQ-editor's `show_object` helper. Provide a no-op
    # stand-in so those scripts execute without modification.
    def _noop_show_object(*args, **kwargs):
        return None

    module.__dict__.setdefault("show_object", _noop_show_object)
    spec.loader.exec_module(module)  # type: ignore[misc]
    return module


def resolve_model_object(module):
    """
    Retrieve the CAD object or builder callable from the generated module.

    Expected exported symbols (checked in order):
    - build(): should construct and return the CAD object.
    - main(): optional alias.
    - MODEL/model/result/RESULT: already instantiated CAD objects.
    """
    if hasattr(module, "build") and callable(module.build):
        return module.build()
    if hasattr(module, "main") and callable(module.main):
        return module.main()
    for attr_name in ("MODEL", "model", "result", "RESULT"):
        if hasattr(module, attr_name):
            return getattr(module, attr_name)
    raise AttributeError(
        "Generated module must expose a callable `build()` or `main()` returning a CAD object."
    )


def as_shape(obj):
    """
    Convert the generated CAD object into a cadquery.Shape for downstream exports.
    """
    import cadquery as cq

    if isinstance(obj, cq.Workplane):
        return obj.val()
    if isinstance(obj, cq.Assembly):
        return obj.toCompound()
    if isinstance(obj, cq.Shape):
        return obj
    if hasattr(obj, "shape"):
        shape = obj.shape
        if isinstance(shape, cq.Shape):
            return shape
    raise TypeError(
        "Unable to convert generated object to a cadquery.Shape; ensure build() returns Workplane/Assembly."
    )


def export_stl(shape, stl_path: Path) -> None:
    """
    Export the cadquery.Shape to an STL file.
    """
    from cadquery import exporters

    stl_path.parent.mkdir(parents=True, exist_ok=True)
    exporters.export(shape, str(stl_path), exportType="STL")


def render_multiviews(stl_path: Path, image_dir: Path, views: Iterable[str]) -> List[Path]:
    """
    Render the STL into multi-view PNGs and return the saved file paths.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    import trimesh

    mesh = trimesh.load_mesh(str(stl_path), force="mesh")
    if mesh.is_empty:  # pragma: no cover - defensive
        raise ValueError("Rendered mesh is empty.")

    triangles = mesh.triangles
    bounds = mesh.bounds
    min_bounds = bounds[0]
    max_bounds = bounds[1]
    center = (min_bounds + max_bounds) / 2.0
    max_range = float(np.max(max_bounds - min_bounds))

    def _plot(ax):
        collection = Poly3DCollection(triangles, linewidths=0.05, alpha=0.85)
        collection.set_facecolor("#89b4fa")
        collection.set_edgecolor("#1f2335")
        ax.add_collection3d(collection)
        ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
        ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
        ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
        ax.set_axis_off()

    view_angles = {
        "front": (0.0, 0.0),
        "side": (0.0, 90.0),
        "top": (90.0, 0.0),
        "iso": (35.0, 45.0),
    }

    image_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for view_name in views:
        elev, azim = view_angles.get(view_name, (35.0, 45.0))
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111, projection="3d")
        _plot(ax)
        ax.view_init(elev=elev, azim=azim)
        output_path = image_dir / f"{view_name}.png"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0, transparent=True)
        plt.close(fig)
        saved.append(output_path.resolve())
    return saved


def run_pipeline(
    *,
    script_path: Path,
    stl_path: Path,
    image_dir: Path,
    views: Iterable[str],
) -> list[Path]:
    """
    Execute the CAD generation script and produce STL + multi-view images.

    Returns a list of image paths pointing to the rendered views.
    """
    module = load_generated_module(script_path)
    cad_object = resolve_model_object(module)
    shape = as_shape(cad_object)
    export_stl(shape, stl_path)
    return render_multiviews(stl_path, image_dir, views)

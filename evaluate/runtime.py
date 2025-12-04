from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluate.geometry import run_pipeline  # type: ignore[import]
else:
    from .geometry import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime executor for generated CAD code.")
    parser.add_argument("--cad-script", type=Path, required=True, help="Path to the generated CAD python script.")
    parser.add_argument("--stl-output", type=Path, required=True, help="Where to write the STL artefact.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory for multi-view images.")
    parser.add_argument(
        "--views",
        type=str,
        default="front,side,top,iso",
        help="Comma separated list of views to render (front,side,top,iso).",
    )
    args = parser.parse_args()

    try:
        views = [view.strip() for view in args.views.split(",") if view.strip()]
        image_paths = run_pipeline(
            script_path=args.cad_script,
            stl_path=args.stl_output,
            image_dir=args.image_dir,
            views=views,
        )
        payload = {
            "success": True,
            "stl_path": str(args.stl_output.resolve()),
            "image_paths": [str(path) for path in image_paths],
        }
        print(json.dumps(payload))
        return 0
    except Exception as exc:  # pragma: no cover - exercised only on failure
        payload = {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(payload))
        return 1


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())

"""Export the FastAPI OpenAPI schema to doc/openapi.json.

Run from anywhere (the backend package path is resolved relative to this
file, not to the process's current working directory):

    uv run --project backend python scripts/export_openapi.py

Importing `app.main` builds the real `FastAPI` app via `create_app()`,
which also runs `setup_tracing()` and (if the module is present) sets up
logging -- both are best-effort and never raise even if Postgres/Ollama/
Phoenix are unreachable, so this script works with or without the stack
running. It only reads the generated OpenAPI schema; it makes no requests
against a running backend.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.main import app  # noqa: E402

out = ROOT / "doc" / "openapi.json"
out.parent.mkdir(parents=True, exist_ok=True)
schema = app.openapi()
out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
n_paths = len(schema.get("paths", {}))
print(f"wrote {out} ({n_paths} paths)")

"""The playground web server.

This file is itself lab 07's subject matter: a real FastAPI application with
validated inputs, typed responses and generated documentation. Once it is
running, http://127.0.0.1:8000/docs gives you an interactive page built
automatically from the type hints below -- try the endpoints there, then come
back and read how they are declared.

Start it with:
    python -m uvicorn app.main:app --reload

--reload watches the source and restarts on save, which is what you want
while learning. Never use it in production.

SECURITY NOTE
-------------
/api/scratch executes Python you type, in a subprocess, on your machine. That
is the entire point of a scratchpad, and it is completely safe while the
server is bound to 127.0.0.1 -- nothing outside your computer can reach it.
Do NOT bind this server to 0.0.0.0, put it behind a tunnel, or deploy it. The
run script pins the host for exactly this reason.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.runner import run_lab_tests
from labs import DATA_DIR, load_clean_orders, load_customers
from labs.lab07_api_build import CategoryQuery, RegionQuery, run_category_query, run_region_query
from labs.registry import LABS, LABS_BY_ID

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Python Data Playground",
    description=(
        "A learning environment for Python data analytics and APIs. "
        "Every endpoint below is a working example of the patterns the labs teach."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Caching the datasets
#
# Reading three CSVs on every request would make the UI feel sluggish for no
# reason -- the files do not change while the server runs. This is the
# simplest possible cache: a module-level dict, populated on first use.
# (If you regenerate the data, restart the server.)
# ---------------------------------------------------------------------------

_cache: dict[str, pd.DataFrame] = {}


def orders() -> pd.DataFrame:
    if "orders" not in _cache:
        _cache["orders"] = load_clean_orders()
    return _cache["orders"]


def customers() -> pd.DataFrame:
    if "customers" not in _cache:
        _cache["customers"] = load_customers()
    return _cache["customers"]


# ---------------------------------------------------------------------------
# Lab catalogue and test running
# ---------------------------------------------------------------------------


@app.get("/api/labs", tags=["labs"])
def list_labs() -> list[dict]:
    """The lab catalogue, in order."""
    return [lab.to_dict() for lab in LABS]


@app.get("/api/labs/{lab_id}/source", tags=["labs"])
def lab_source(lab_id: str) -> dict:
    """The full source of a lab module, for reading in the browser."""
    lab = LABS_BY_ID.get(lab_id)
    if lab is None:
        raise HTTPException(status_code=404, detail=f"No lab {lab_id!r}")

    module_path = REPO_ROOT / (lab.module.replace(".", "/") + ".py")
    test_path = REPO_ROOT / lab.test_file
    return {
        "lab_id": lab_id,
        "module_path": str(module_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source": module_path.read_text(encoding="utf-8"),
        "test_path": lab.test_file,
        "tests": test_path.read_text(encoding="utf-8"),
    }


class SourceUpdate(BaseModel):
    """A new version of a lab module, sent from the browser editor."""

    source: str = Field(max_length=500_000)


@app.put("/api/labs/{lab_id}/source", tags=["labs"])
def save_lab_source(lab_id: str, update: SourceUpdate) -> dict:
    """Overwrite a lab module with edited source.

    Only the eight modules in the registry can be written, and only those --
    `lab_id` is looked up in LABS_BY_ID rather than turned into a path, so
    there is no filename here for a caller to manipulate.

    Note what is NOT editable: the test files. That is deliberate. The tests
    are the specification, and an exercise you can edit the grader for is not
    an exercise. If you think a test is wrong, say so -- but fix it in your
    editor with intent, not by accident at 1am.

    Your safety net is git. `git diff labs/` shows everything you have
    changed, and `git checkout labs/lab01_loading.py` puts a file back the
    way it shipped.
    """
    lab = LABS_BY_ID.get(lab_id)
    if lab is None:
        raise HTTPException(status_code=404, detail=f"No lab {lab_id!r}")

    module_path = REPO_ROOT / (lab.module.replace(".", "/") + ".py")

    # Compile before writing so we can hand back a precise location. We still
    # save a file that does not compile -- half-finished code is a normal
    # state to be in, and a tool that refuses to save it is a tool that loses
    # your work. Report the problem; do not withhold the save.
    syntax_error = None
    try:
        compile(update.source, str(module_path), "exec")
    except SyntaxError as exc:
        syntax_error = {
            "message": exc.msg,
            "line": exc.lineno,
            "offset": exc.offset,
            "text": (exc.text or "").rstrip(),
        }

    # newline="\n" keeps the file LF on Windows, matching .gitattributes --
    # otherwise every browser save would show up as a whole-file diff.
    module_path.write_text(update.source, encoding="utf-8", newline="\n")

    return {
        "saved": True,
        "path": str(module_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "bytes": len(update.source.encode("utf-8")),
        "syntax_error": syntax_error,
    }


@app.post("/api/labs/{lab_id}/run", tags=["labs"])
def run_lab(lab_id: str) -> dict:
    """Run a lab's pytest suite and return a structured result."""
    lab = LABS_BY_ID.get(lab_id)
    if lab is None:
        raise HTTPException(status_code=404, detail=f"No lab {lab_id!r}")
    return run_lab_tests(lab.test_file)


# ---------------------------------------------------------------------------
# Dataset browsing
# ---------------------------------------------------------------------------


@app.get("/api/datasets", tags=["data"])
def list_datasets() -> list[dict]:
    """Every CSV in data/, with its shape and columns."""
    out = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        frame = pd.read_csv(path, nrows=5)
        total_rows = sum(1 for _ in path.open(encoding="utf-8")) - 1  # minus header
        out.append(
            {
                "name": path.name,
                "rows": total_rows,
                "columns": list(frame.columns),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return out


@app.get("/api/datasets/{name}", tags=["data"])
def preview_dataset(
    name: str,
    limit: int = Query(default=25, ge=1, le=500, description="Rows to return."),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """A page of rows from a dataset, plus its dtypes and missing-value counts.

    Note the Query(ge=, le=) constraints -- the same validation idea as lab
    07, applied to query parameters rather than a model. Without the le=500
    ceiling, a caller could ask for every row of a very large file and stall
    the server.
    """
    path = DATA_DIR / name
    # Guard against path traversal: `name` comes from the URL, so a caller
    # could send "../../secrets.txt". Checking that the resolved path is still
    # inside DATA_DIR is the standard defence. Never trust a filename from a
    # request, even on a localhost tool.
    if not path.resolve().is_relative_to(DATA_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail=f"No dataset {name!r}")

    frame = pd.read_csv(path)
    page = frame.iloc[offset : offset + limit]

    return {
        "name": name,
        "total_rows": int(len(frame)),
        "offset": offset,
        "columns": list(frame.columns),
        "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "missing": {col: int(count) for col, count in frame.isna().sum().items()},
        # to_json then back gives NaN -> null, which raw dicts do not: NaN is
        # not valid JSON, and some clients choke on it.
        "rows": page.where(pd.notna(page), None).to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Live lab-07 endpoints
#
# These are the real thing: the same functions the tests exercise, served over
# HTTP. The category endpoint works now; the region endpoint starts returning
# data the moment you finish the exercise. Open /docs and try them.
# ---------------------------------------------------------------------------


@app.get("/api/analytics/categories", tags=["analytics"])
def categories_endpoint(query: Annotated[CategoryQuery, Query()]) -> list[dict]:
    """Revenue by category. Lab 07's worked example, live."""
    return run_category_query(orders(), query)


@app.get("/api/analytics/regions", tags=["analytics"])
def regions_endpoint(query: Annotated[RegionQuery, Query()]) -> list[dict]:
    """Revenue by region. Lab 07's exercise -- returns 501 until you write it."""
    try:
        return run_region_query(orders(), customers(), query)
    except NotImplementedError as exc:
        # 501 Not Implemented is the honest status code here, and it is more
        # useful to the caller than letting the exception become a 500.
        raise HTTPException(status_code=501, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# The scratchpad
# ---------------------------------------------------------------------------


class Snippet(BaseModel):
    """A chunk of Python to execute in a fresh subprocess."""

    code: str = Field(max_length=20_000)
    timeout: int = Field(default=20, ge=1, le=60)


@app.post("/api/scratch", tags=["scratch"])
def run_scratch(snippet: Snippet) -> dict:
    """Execute Python and return whatever it printed.

    Runs in a separate process with the repo root on sys.path, so `from labs
    import load_clean_orders` works exactly as it does in a test. A timeout
    stops a runaway loop from pinning a core forever.

    This is deliberately NOT sandboxed -- the code runs with your permissions,
    like any script you would run yourself. See the SECURITY NOTE at the top
    of this file: keep the server on 127.0.0.1.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippet.py"
        # A preamble so imports resolve and pandas prints readable tables.
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "import pandas as pd\n"
            "pd.set_option('display.width', 120)\n"
            "pd.set_option('display.max_columns', 40)\n\n"
            + snippet.code,
            encoding="utf-8",
        )

        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=snippet.timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"Timed out after {snippet.timeout}s -- check for an infinite loop.",
                "exit_code": None,
            }

    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
    }


# ---------------------------------------------------------------------------
# Static front end
#
# Mounted last so that the /api routes above win when paths could overlap.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def no_cache_static(request, call_next):
    """Stop the browser caching the front end.

    Without this, editing app/static/app.js and reloading can serve you the
    previous version from cache -- your change is on disk, the page does not
    have it, and you spend twenty minutes debugging code that is not running.

    A real deployment wants the opposite (long cache lifetimes plus hashed
    filenames), but this is a local tool you are actively editing, and
    correctness beats a few milliseconds here.
    """
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

import os
import sys
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DOCS_ENDPOINT = "https://api.esios.ree.es/documents"
NAME_PREFIX = "ERNI Nudos RR.TT TNP_"
LOCALE = "es"

OUT_DIR = Path(".")
STATE_PATH = OUT_DIR / "erni_rrtt_state.json"


def prev_month_yyyymm(today: date) -> str:
    year = today.year
    month = today.month - 1
    if month == 0:
        month = 12
        year -= 1
    return f"{year}{month:02d}"


def build_headers(api_key: str) -> Dict[str, str]:
    # ESIOS is sometimes used with either x-api-key or Authorization Token.
    # We set both to be robust; whichever the server expects will work.
    return {
        "Accept": "application/json",
        "x-api-key": api_key,
        "Authorization": f'Token token="{api_key}"',
    }


def fetch_documents(session: requests.Session) -> List[Dict[str, Any]]:
    # Many ESIOS endpoints accept locale param; keep it explicit.
    r = session.get(DOCS_ENDPOINT, params={"locale": LOCALE}, timeout=60)
    if r.status_code == 401:
        raise RuntimeError("401 Unauthorized. Check that the API_KEY secret is valid and has access.")
    r.raise_for_status()

    data = r.json()

    # Your exported file looks like a plain JSON array of objects.
    # But some APIs wrap it. Support both.
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("data", "items", "results"):
            if k in data and isinstance(data[k], list):
                return data[k]

    raise RuntimeError("Unexpected documents response shape; couldn't find a list of documents.")


def find_target_doc(docs: List[Dict[str, Any]], yyyymm: str) -> Optional[Dict[str, Any]]:
    target_name = f"{NAME_PREFIX}{yyyymm}"
    for d in docs:
        name = str(d.get("name", "")).strip()
        if name == target_name:
            # Prefer truly downloadable PDFs
            if str(d.get("type", "")).lower() == "pdf" and bool(d.get("downloable", True)):
                return d
            return d
    return None


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def download_file(session: requests.Session, url: str, out_path: Path) -> None:
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)


def main() -> None:
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        print("Missing API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    yyyymm = prev_month_yyyymm(date.today())

    session = requests.Session()
    session.headers.update(build_headers(api_key))

    docs = fetch_documents(session)
    doc = find_target_doc(docs, yyyymm)

    if not doc:
        print(f"Not found yet: {NAME_PREFIX}{yyyymm}. No download performed.")
        # Exit 0: not an error; just not published (or listing changed)
        sys.exit(0)

    url = str(doc.get("url", "")).strip()
    if not url:
        raise RuntimeError(f"Found document {doc.get('name')} but it has no 'url' field.")

    out_pdf = OUT_DIR / f"ERNI_Nudos_RRTT_TNP_{yyyymm}.pdf"

    state = load_state()
    already = state.get("downloaded", {}).get(yyyymm)
    if already and Path(already).exists():
        print(f"Already downloaded for {yyyymm}: {already}")
        sys.exit(0)

    download_file(session, url, out_pdf)

    state.setdefault("downloaded", {})[yyyymm] = str(out_pdf)
    state["last_run"] = date.today().isoformat()
    state["last_target"] = yyyymm
    state["last_doc_id"] = doc.get("id")
    save_state(state)

    print(f"Downloaded successfully: {out_pdf.name}")


if __name__ == "__main__":
    main()

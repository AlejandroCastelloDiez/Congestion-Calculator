from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Literal
from urllib.parse import urljoin
import argparse
import json
import math
import os
import re
import unicodedata

import pandas as pd
import requests
from difflib import SequenceMatcher
from pypdf import PdfReader

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
PDF_DIR = OUT_DIR / "pdfs"

BASE_URL = "https://api.esios.ree.es"
DOCS_URL = f"{BASE_URL}/es/documentations"

COL_NIRE = "% NIRE over Peninsula PDBF (Province)"
COL_CURT = "Curtailment Ratio (Province)"

VALUE_COL_1 = "% NIRE over Peninsula PDBF (Province)"
VALUE_COL_2 = "Curtailment Ratio (Province)"


def esios_headers(api_key: str) -> Dict[str, str]:
    return {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "User-Agent": "Mozilla/5.0",
    }


def is_pdf_media(m: Dict[str, Any]) -> bool:
    if str(m.get("media_type", "")).lower() != "document":
        return False
    ct = str(m.get("document_value_content_type", "")).lower()
    fn = str(m.get("document_value_file_name", "")).lower()
    return ("pdf" in ct) or fn.endswith(".pdf")


def media_download_url(m: Dict[str, Any]) -> Optional[str]:
    rel = m.get("download")
    if not rel:
        return None
    rel = str(rel).strip()
    return rel if rel.startswith("http") else urljoin(BASE_URL, rel)


def get_page(session: requests.Session, page: int, per_page: int = 50) -> Dict[str, Any]:
    params = {"locale": "es", "page": page, "per_page": per_page, "order": "published"}
    r = session.get(DOCS_URL, params=params, timeout=60)
    if r.status_code == 401:
        raise RuntimeError("401 Unauthorized. Check API_KEY.")
    if r.status_code == 404:
        raise RuntimeError(f"404 Not Found: {r.url} (endpoint should be /es/documentations).")
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict) or "contents" not in data:
        raise RuntimeError("Unexpected response shape from /es/documentations.")
    return data


def iter_documentations(session: requests.Session, max_pages: int = 25, per_page: int = 50):
    for page in range(1, max_pages + 1):
        data = get_page(session, page=page, per_page=per_page)
        contents = data.get("contents") or []
        if not contents:
            return
        yield from contents
        if len(contents) < per_page:
            return


def best_pdf_media(item: Dict[str, Any], yyyymm: str) -> Optional[Dict[str, Any]]:
    media = item.get("media") or []
    pdfs = [m for m in media if is_pdf_media(m) and media_download_url(m)]
    if not pdfs:
        return None

    def mscore(m: Dict[str, Any]) -> int:
        t = (str(m.get("title", "")) + " " + str(m.get("document_value_file_name", ""))).upper()
        s = 0
        if yyyymm in t:
            s += 10
        if "ERNI" in t:
            s += 4
        if "PDBF" in t:
            s += 4
        if "FASE I" in t:
            s += 4
        if "RR.TT" in t or "RRTT" in t or "RR TT" in t:
            s += 2
        return s

    pdfs.sort(key=mscore, reverse=True)
    return pdfs[0]


def download_stream(session: requests.Session, url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)


def download_erni_pdbf_fase_i(api_key: str, year: int, month: int, *, max_pages: int = 25) -> Path | None:
    if not (1 <= month <= 12):
        raise ValueError("month must be 1..12")
    if not (2000 <= year <= 2100):
        raise ValueError("year out of expected range")

    yyyymm = f"{year}{month:02d}"
    target_title = f"ERNI Nudos RR.TT PDBF Fase I_{yyyymm}"
    out_pdf = PDF_DIR / f"ERNI_RRTT_PDBF_FaseI_{yyyymm}.pdf"

    if out_pdf.exists() and out_pdf.stat().st_size > 0:
        return out_pdf

    s = requests.Session()
    s.headers.update(esios_headers(api_key))

    seen_ids = set()
    for item in iter_documentations(s, max_pages=max_pages, per_page=50):
        _id = item.get("id")
        if _id in seen_ids:
            continue
        seen_ids.add(_id)

        title = str(item.get("title", "")).strip()
        if title != target_title:
            continue

        m = best_pdf_media(item, yyyymm)
        if not m:
            return None

        dl_url = media_download_url(m)
        if not dl_url:
            return None

        download_stream(s, dl_url, out_pdf)
        return out_pdf

    return None


def pdf_parser(year: int, month: int) -> pd.DataFrame:
    yyyymm = f"{year:04d}{month:02d}"
    pdf_path = PDF_DIR / f"ERNI_RRTT_PDBF_FaseI_{yyyymm}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    reader = PdfReader(str(pdf_path))
    if len(reader.pages) < 3:
        raise RuntimeError(f"PDF has {len(reader.pages)} pages; expected at least 3.")
    page = reader.pages[2]
    text = page.extract_text() or ""

    spaced_out_re = re.compile(r"(?:\b[A-ZÁÉÍÓÚÜÑ0-9]\b\s+){6,}")

    def normalize_line(line: str, force_fix: bool = False) -> str:
        line = line.rstrip("\n")
        if not line.strip():
            return ""
        if force_fix or spaced_out_re.search(line):
            sentinel = "\u0000"
            s = re.sub(r"\s{2,}", sentinel, line.strip())
            s = re.sub(r"\s+", "", s)
            s = s.replace(sentinel, " ")
            return re.sub(r"\s+", " ", s).strip()
        return re.sub(r"\s+", " ", line).strip()

    def normalize_node_name(name: str) -> str:
        return re.sub(r"\s+", "", (name or "").strip()).upper()

    split_re = re.compile(r"^(?P<name>.+?)\s+(?P<num>\d.*)$")

    rest_re_strict = re.compile(
        r"^(?P<pdbf>\d+,\d{2})"
        r"(?P<nire>\d+,\d{2})"
        r"(?P<pct_node>\d+,\d{2})%"
        r"(?P<pct_pen>\d+,\d{2})%$"
    )
    rest_re_relaxed = re.compile(
        r"^(?P<pdbf>\d+,\d{2})"
        r"(?P<nire>\d+,\d{2})"
        r"(?P<pct_node>\d+,\d{2})%"
        r"(?P<pct_pen>\d+,\d{2,3})%$"
    )

    KV_CANDIDATES = ["400", "220", "132", "110", "66", "500", "150"]

    def to_float_es(s: str) -> float:
        return float(s.replace(".", "").replace(",", "."))

    def parse_numeric_blob(num_blob: str, rest_re: re.Pattern):
        blob = re.sub(r"\s+", "", num_blob)
        for kv in KV_CANDIDATES:
            if not blob.startswith(kv):
                continue
            rest = blob[len(kv) :]
            m = rest_re.match(rest)
            if not m:
                continue
            return {
                "kv": kv,
                "pdbf": to_float_es(m.group("pdbf")),
                "nire": to_float_es(m.group("nire")),
                "pct_node": to_float_es(m.group("pct_node")),
                "pct_pen": to_float_es(m.group("pct_pen")),
            }
        return None

    def run_parse(rest_re: re.Pattern, force_fix: bool) -> pd.DataFrame:
        rows = []
        for raw_line in text.splitlines():
            line = normalize_line(raw_line, force_fix=force_fix)
            if "%" not in line or "," not in line:
                continue
            m = split_re.match(line)
            if not m:
                continue
            name = normalize_node_name(m.group("name"))
            parsed = parse_numeric_blob(m.group("num"), rest_re=rest_re)
            if not parsed:
                continue
            rows.append(
                {
                    "Substation Node": f"{name} {parsed['kv']}",
                    "PDBF Program (GWh)": parsed["pdbf"],
                    "NIRE (GWh)": parsed["nire"],
                    "% NIRE per Substation Node": parsed["pct_node"],
                    "% NIRE over Peninsula PDBF": parsed["pct_pen"],
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "Substation Node",
                "PDBF Program (GWh)",
                "NIRE (GWh)",
                "% NIRE per Substation Node",
                "% NIRE over Peninsula PDBF",
            ],
        )

    df = run_parse(rest_re_strict, force_fix=False)
    if df.empty:
        df = run_parse(rest_re_relaxed, force_fix=False)
    if df.empty:
        df = run_parse(rest_re_relaxed, force_fix=True)
    return df


def fix_mojibake(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    if "Ã" not in s and "Â" not in s and "�" not in s:
        return s
    for enc in ("latin1", "cp1252"):
        try:
            return s.encode(enc, errors="strict").decode("utf-8", errors="strict")
        except Exception:
            pass
    return s


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def canon_name(s: str) -> str:
    s = fix_mojibake(s)
    s = strip_accents(str(s).upper())
    return re.sub(r"[^A-Z0-9]", "", s)


def split_node_and_kv(s: str):
    s = fix_mojibake(s)
    s = str(s).strip().upper()
    m = re.search(r"(\d{2,4})\s*$", s)
    if m:
        kv = m.group(1)
        name = s[: m.start(1)].strip()
        return name, kv
    return s, None


ABBREV_MAP = {
    "PTO": "PUERTO",
    "STA": "SANTA",
    "STO": "SANTO",
    "SN": "SAN",
    "STA.": "SANTA",
    "SN.": "SAN",
}

CONNECTORS = {"DE", "DEL", "LA", "LAS", "LOS", "EL", "Y"}


def expand_abbrev_tokens(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        t_clean = re.sub(r"[^A-Z0-9ÁÉÍÓÚÜÑ\.]", "", t)
        out.append(ABBREV_MAP.get(t_clean, t_clean))
    return out


def normalize_name_variants(name_part: str) -> tuple[str, str]:
    s = fix_mojibake(name_part).strip().upper()
    tokens = re.split(r"\s+", s)
    tokens = [t for t in tokens if t]
    tokens = expand_abbrev_tokens(tokens)
    full_tokens = tokens
    reduced_tokens = [t for t in tokens if t not in CONNECTORS]
    full = re.sub(r"\s+", "", " ".join(full_tokens)).strip()
    reduced = re.sub(r"\s+", "", " ".join(reduced_tokens)).strip()
    return full, reduced


def node_key_from_text(s: str) -> str:
    name_part, kv = split_node_and_kv(s)
    full, _ = normalize_name_variants(name_part)
    if kv:
        return f"{full} {kv}"
    return full


_vowels_re = re.compile(r"[AEIOU]")


def consonant_skeleton(canon: str) -> str:
    return _vowels_re.sub("", canon)


def is_subsequence(shorter: str, longer: str) -> bool:
    it = iter(longer)
    return all(ch in it for ch in shorter)


def subseq_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if len(a) <= len(b) and is_subsequence(a, b):
        return len(a) / len(b)
    if len(b) < len(a) and is_subsequence(b, a):
        return len(b) / len(a)
    return 0.0


def base_score(q: str, c: str) -> float:
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    same_first = q[0] == c[0]
    if same_first and (c.startswith(q) or q.startswith(c)):
        return 0.97
    if same_first and (q in c or c in q):
        return 0.94
    if same_first:
        sr = subseq_ratio(q, c)
        if sr >= 0.45:
            t = min(1.0, max(0.0, (sr - 0.45) / 0.55))
            return 0.90 + 0.085 * t
    return SequenceMatcher(None, q, c).ratio()


def score_match_multi(q_full: str, q_red: str, c_full: str, c_red: str) -> float:
    pairs = [(q_full, c_full), (q_full, c_red), (q_red, c_full), (q_red, c_red)]
    best = 0.0
    for q, c in pairs:
        if not q or not c:
            continue
        best = max(best, base_score(q, c))
        if q[0] != c[0]:
            continue
        qsk = consonant_skeleton(q)
        csk = consonant_skeleton(c)
        if qsk and csk and (qsk in csk or csk in qsk):
            best = max(best, 0.93)
        elif qsk and csk:
            skr = subseq_ratio(qsk, csk)
            if skr >= 0.50:
                t = min(1.0, max(0.0, (skr - 0.50) / 0.50))
                best = max(best, 0.90 + 0.06 * t)
    return best


def province_aggregation_from_pdf_light(
    df_pdf: pd.DataFrame,
    excel_path: str | Path,
    similarity_cutoff: float = 0.90,
    debug: bool = False,
    min_candidate_len: int = 6,
    min_candidate_ratio: float = 0.70,
    as_percent: bool = True,
    percent_decimals: int = 2,
) -> pd.DataFrame:
    need = {"Substation Node", "PDBF Program (GWh)", "NIRE (GWh)", "% NIRE over Peninsula PDBF"}
    missing = need - set(df_pdf.columns)
    if missing:
        raise ValueError(f"df_pdf missing columns: {sorted(missing)}")

    work = df_pdf.copy()
    work["Substation Node"] = work["Substation Node"].apply(fix_mojibake)

    work["PDBF Program (GWh)"] = pd.to_numeric(work["PDBF Program (GWh)"], errors="coerce")
    work["NIRE (GWh)"] = pd.to_numeric(work["NIRE (GWh)"], errors="coerce")
    work["% NIRE over Peninsula PDBF"] = pd.to_numeric(work["% NIRE over Peninsula PDBF"], errors="coerce")

    def _q_variants(node: str):
        name_part, kv = split_node_and_kv(node)
        q_full, q_red = normalize_name_variants(name_part)
        return pd.Series([q_full, q_red, kv, canon_name(q_full), canon_name(q_red)])

    work[["__q_full__", "__q_red__", "__kv__", "__q_full_c__", "__q_red_c__"]] = work["Substation Node"].apply(_q_variants)
    work["__node_key__"] = work["Substation Node"].apply(node_key_from_text)

    map_df = pd.read_excel(excel_path, dtype=str)
    need_map = {"Node Name", "Province"}
    missing_map = need_map - set(map_df.columns)
    if missing_map:
        raise ValueError(f"Excel missing columns: {sorted(missing_map)}")

    map_df = map_df.dropna(subset=["Node Name", "Province"]).copy()
    map_df["Node Name"] = map_df["Node Name"].apply(fix_mojibake)

    def _c_variants(node: str):
        name_part, kv = split_node_and_kv(node)
        c_full, c_red = normalize_name_variants(name_part)
        return pd.Series([kv, canon_name(c_full), canon_name(c_red)])

    map_df[["__kv__", "__c_full_c__", "__c_red_c__"]] = map_df["Node Name"].apply(_c_variants)
    map_df["__node_key__"] = map_df["Node Name"].apply(node_key_from_text)

    candidates_by_kv: dict[str, list[dict]] = {}
    for _, r in map_df.iterrows():
        kv = r["__kv__"]
        if pd.isna(kv) or not str(kv).strip():
            continue
        candidates_by_kv.setdefault(str(kv), []).append(
            {"prov": r["Province"], "key": r["__node_key__"], "full_c": r["__c_full_c__"], "red_c": r["__c_red_c__"]}
        )

    kv_sizes = {k: len(v) for k, v in candidates_by_kv.items()}

    def best_match(q_full_c: str, q_red_c: str, kv: str):
        if not kv or (not q_full_c and not q_red_c):
            return (None, None, None, "no_kv_or_name")

        cands = candidates_by_kv.get(str(kv), [])
        if not cands:
            return (None, None, None, "no_candidates_for_kv")

        qcanon = q_full_c if len(q_full_c) >= len(q_red_c) else q_red_c
        qlen = max(1, len(qcanon))

        filtered = []
        for c in cands:
            if c["full_c"] in (q_full_c, q_red_c) or c["red_c"] in (q_full_c, q_red_c):
                filtered.append(c)
                continue

            clen = max(len(c["full_c"]), len(c["red_c"]))
            if clen < int(min_candidate_len):
                continue
            if (clen / qlen) < float(min_candidate_ratio):
                continue
            filtered.append(c)

        if not filtered:
            return (None, None, None, "filtered_all_too_short")

        best = None
        best_score = -1.0
        for c in filtered:
            sc = score_match_multi(q_full_c, q_red_c, c["full_c"], c["red_c"])
            if sc > best_score:
                best_score = sc
                best = c

        if best_score >= similarity_cutoff:
            return (best["prov"], best["key"], float(best_score), "matched")

        return (None, best["key"] if best else None, float(best_score) if best else None, "below_cutoff")

    out = work.apply(lambda r: best_match(r["__q_full_c__"], r["__q_red_c__"], r["__kv__"]), axis=1)
    work["Province"] = out.apply(lambda x: x[0])
    work["Matched Node Key"] = out.apply(lambda x: x[1])
    work["Match Score"] = out.apply(lambda x: x[2])
    work["Match Status"] = out.apply(lambda x: x[3])

    if debug:
        print("Candidate counts by KV:", kv_sizes)
        unmatched = work[work["Province"].isna()][["Substation Node", "__node_key__", "__kv__", "Matched Node Key", "Match Score", "Match Status"]]
        if not unmatched.empty:
            print(unmatched.sort_values(["Match Status", "Match Score", "Substation Node"], ascending=[True, False, True]).to_string(index=False))

    agg = (
        work.dropna(subset=["Province"])
        .groupby("Province", as_index=False)
        .agg(
            **{
                "% NIRE over Peninsula PDBF (Province)": ("% NIRE over Peninsula PDBF", "sum"),
                "_sum_nire": ("NIRE (GWh)", "sum"),
                "_sum_pdbf": ("PDBF Program (GWh)", "sum"),
            }
        )
    )

    agg["Curtailment Ratio (Province)"] = agg["_sum_nire"] / agg["_sum_pdbf"]
    if as_percent:
        agg["Curtailment Ratio (Province)"] = (agg["Curtailment Ratio (Province)"] * 100).round(percent_decimals)

    agg = agg.drop(columns=["_sum_nire", "_sum_pdbf"]).sort_values("% NIRE over Peninsula PDBF (Province)", ascending=False, ignore_index=True)
    return agg


def store_province_metrics_light_json(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    year: int,
    month: int,
    mode: Literal["append", "overwrite"] = "append",
    sort_desc_by: str = VALUE_COL_1,
    allow_overwrite_snapshot: bool = False,
) -> Path:
    out_path = Path(out_path)
    snapshot_key = f"{int(year):04d}-{int(month):02d}"

    required = {"Province", VALUE_COL_1, VALUE_COL_2}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df must contain columns: {sorted(required)}. Missing: {sorted(missing)}")

    data = df[["Province", VALUE_COL_1, VALUE_COL_2]].copy()
    data["Province"] = data["Province"].astype(str).str.strip()
    data = data[data["Province"] != ""]

    def to_num(s: pd.Series) -> pd.Series:
        x = s.astype(str).str.replace(" ", "", regex=False).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
        return pd.to_numeric(x, errors="coerce")

    data[VALUE_COL_1] = to_num(data[VALUE_COL_1])
    data[VALUE_COL_2] = to_num(data[VALUE_COL_2])
    data = data.dropna(subset=[VALUE_COL_1, VALUE_COL_2], how="all")

    if sort_desc_by not in (VALUE_COL_1, VALUE_COL_2, "Province", None):
        raise ValueError(f"sort_desc_by must be one of: {VALUE_COL_1!r}, {VALUE_COL_2!r}, 'Province', or None")

    if sort_desc_by is not None:
        if sort_desc_by == "Province":
            data = data.sort_values("Province", ascending=True, kind="mergesort")
        else:
            data = data.sort_values(sort_desc_by, ascending=False, kind="mergesort")

    month_payload = {}
    for _, r in data.iterrows():
        prov = r["Province"]
        month_payload[prov] = {
            VALUE_COL_1: (None if pd.isna(r[VALUE_COL_1]) else float(r[VALUE_COL_1])),
            VALUE_COL_2: (None if pd.isna(r[VALUE_COL_2]) else float(r[VALUE_COL_2])),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "overwrite":
        payload = {snapshot_key: month_payload}
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {}
    else:
        existing = {}

    if (snapshot_key in existing) and (not allow_overwrite_snapshot):
        return out_path

    existing[snapshot_key] = month_payload
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def canon(s):
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


def top_suggestions(bad_key: str, keys, n=5):
    scored = sorted(((k, SequenceMatcher(None, bad_key, k).ratio()) for k in keys), key=lambda x: x[1], reverse=True)
    return scored[:n]


def nice_ceiling(x: float) -> float:
    if x <= 0 or not math.isfinite(x):
        return 1.0
    exp = math.floor(math.log10(x))
    frac = x / (10**exp)
    if frac <= 1:
        nice = 1
    elif frac <= 2:
        nice = 2
    elif frac <= 2.5:
        nice = 2.5
    elif frac <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10**exp)


def _add_logo(fig, logo_path: str | Path, *, x=0.82, y=0.82, w=0.16, h=0.16, alpha=1.0):
    logo_path = Path(logo_path)
    if not logo_path.exists():
        return
    try:
        img = plt.imread(str(logo_path))
    except Exception:
        return
    ax_logo = fig.add_axes([x, y, w, h], anchor="NE", zorder=10)
    ax_logo.imshow(img, alpha=alpha)
    ax_logo.axis("off")


def plot_heatmap(
    merged_gdf: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    *,
    vmin: float,
    vmax: float,
    legend_label: str,
    cmap,
    logo_path: str | Path,
    out_path: str | Path | None,
    dpi: int = 200,
):
    plot_vals = pd.to_numeric(merged_gdf[value_col], errors="coerce").fillna(0).clip(lower=0, upper=vmax)
    tmp = merged_gdf.copy()
    tmp["_plot_"] = plot_vals

    zero = tmp[tmp["_plot_"] == 0].copy()
    pos = tmp[tmp["_plot_"] > 0].copy()

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    if not zero.empty:
        zero.plot(ax=ax, color="white", linewidth=0.6, edgecolor="black")

    if not pos.empty:
        pos.plot(
            column="_plot_",
            ax=ax,
            cmap=cmap,
            legend=True,
            linewidth=0.6,
            edgecolor="black",
            vmin=vmin,
            vmax=vmax,
            legend_kwds={"shrink": 0.65, "label": legend_label},
        )

    ax.set_title(title, fontsize=14, pad=10)
    ax.axis("off")
    _add_logo(fig, logo_path)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def plot_two_province_heatmaps_from_json(
    json_path: str | Path,
    *,
    year: int,
    month: int,
    vmax_pad_ratio: float = 1.10,
    vmax_floor_nire: float = 0.1,
    vmax_floor_curt: float = 5.0,
    debug_unmatched: bool = False,
    logo_path: str | Path,
    out_dir: str | Path,
    save_png: bool = True,
    dpi: int = 250,
):
    json_path = Path(json_path)
    snapshot_key = f"{int(year):04d}-{int(month):02d}"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    month_obj = payload.get(snapshot_key, None)
    if month_obj is None:
        available = sorted(payload.keys())
        raise KeyError(f"Snapshot '{snapshot_key}' not found. Available: {available[:12]}{'...' if len(available)>12 else ''}")

    rows = []
    for prov, metrics in month_obj.items():
        if not isinstance(metrics, dict):
            continue
        rows.append({"Province": str(prov), COL_NIRE: metrics.get(COL_NIRE, None), COL_CURT: metrics.get(COL_CURT, None)})
    df_prov = pd.DataFrame(rows)
    if df_prov.empty:
        raise ValueError(f"No province rows found for snapshot {snapshot_key}.")

    df_prov[COL_NIRE] = pd.to_numeric(df_prov[COL_NIRE], errors="coerce")
    df_prov[COL_CURT] = pd.to_numeric(df_prov[COL_CURT], errors="coerce")

    nire_max = float(df_prov.loc[df_prov[COL_NIRE] > 0, COL_NIRE].max() or 0.0)
    curt_max = float(df_prov.loc[df_prov[COL_CURT] > 0, COL_CURT].max() or 0.0)

    vmax_nire = max(vmax_floor_nire, nice_ceiling(nire_max * float(vmax_pad_ratio)))
    vmax_curt = max(vmax_floor_curt, nice_ceiling(curt_max * float(vmax_pad_ratio)))

    teal = to_rgba("#219b93ff")
    cmap_teal = LinearSegmentedColormap.from_list("white_to_teal", [(1, 1, 1, 1), teal], N=256)

    gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ESP_2.json"
    geo_path = DATA_DIR / "gadm41_ESP_2.json"
    geo_path.parent.mkdir(parents=True, exist_ok=True)
    if not geo_path.exists():
        import urllib.request
        urllib.request.urlretrieve(gadm_url, str(geo_path))

    gdf = gpd.read_file(str(geo_path))

    gdf["_name1_key_"] = gdf["NAME_1"].apply(canon)
    gdf["_name2_key_"] = gdf["NAME_2"].apply(canon)

    islands_or_excl = (
        gdf["_name1_key_"].str.contains("CANARIAS", na=False)
        | gdf["_name1_key_"].str.contains("BALEARES", na=False)
        | gdf["_name2_key_"].isin({canon("Ceuta"), canon("Melilla")})
    )
    gdf = gdf.loc[~islands_or_excl].copy()

    gdf["_prov_key_"] = gdf["NAME_2"].apply(canon)
    gadm_keys = set(gdf["_prov_key_"].unique())

    alias = {
        canon("Seville"): canon("Sevilla"),
        canon("Navarre"): canon("Navarra"),
        canon("Gipuzkoa"): canon("Guipuzcoa"),
        canon("Bizkaia"): canon("Vizcaya"),
        canon("A Coruña"): canon("A Coruna"),
        canon("A Coruna"): canon("A Coruna"),
        canon("Girona"): canon("Gerona"),
        canon("Lleida"): canon("Lerida"),
        canon("Castelló"): canon("Castellon"),
        canon("Castello"): canon("Castellon"),
        canon("Ciudad Real"): canon("CIUDADREAL"),
        canon("La Rioja"): canon("RIOJA"),
    }

    def choose_key(k0: str) -> str:
        if k0 in gadm_keys:
            return k0
        k1 = alias.get(k0, k0)
        if k1 in gadm_keys:
            return k1
        heur = []
        heur.append(k0.replace(" ", ""))
        heur.append(re.sub(r"^(LA|EL|LOS|LAS)\s+", "", k0).strip())
        heur.append(re.sub(r"^(LA|EL|LOS|LAS)\s+", "", k0).strip().replace(" ", ""))
        for k in heur:
            if k in gadm_keys:
                return k
        return k0

    df = df_prov.copy()
    df["_prov_raw_"] = df["Province"].astype(str)
    df["_prov_key_0_"] = df["_prov_raw_"].apply(canon)
    df["_prov_key_"] = df["_prov_key_0_"].apply(choose_key)

    if debug_unmatched:
        unmatched = sorted(set(df["_prov_key_"].unique()) - gadm_keys)
        if unmatched:
            for u in unmatched:
                sugg = top_suggestions(u, gadm_keys, n=5)
                print(u, sugg)

    merged = gdf.merge(df[["_prov_key_", COL_NIRE, COL_CURT]], on="_prov_key_", how="left")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nire_png = out_dir / f"heatmap_{snapshot_key}_nire.png" if save_png else None
    curt_png = out_dir / f"heatmap_{snapshot_key}_curtailment.png" if save_png else None

    plot_heatmap(
        merged,
        value_col=COL_NIRE,
        title=f"{snapshot_key} — NIRE over Peninsular PDBF (%)",
        vmin=0,
        vmax=vmax_nire,
        legend_label="Percentage over Peninsular PDBF (%)",
        cmap=cmap_teal,
        logo_path=logo_path,
        out_path=nire_png,
        dpi=dpi,
    )

    plot_heatmap(
        merged,
        value_col=COL_CURT,
        title=f"{snapshot_key} — Relative NIRE by Province (%)",
        vmin=0,
        vmax=vmax_curt,
        legend_label="Percentage over Province Constrained Nodes PDBF (%)",
        cmap=cmap_teal,
        logo_path=logo_path,
        out_path=curt_png,
        dpi=dpi,
    )

    return {"nire_png": str(nire_png) if nire_png else None, "curt_png": str(curt_png) if curt_png else None}


def last_complete_month(today: date) -> tuple[int, int]:
    y = today.year
    m = today.month
    if m == 1:
        return (y - 1, 12)
    return (y, m - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month", type=int, default=None)
    parser.add_argument("--similarity-cutoff", type=float, default=0.90)
    parser.add_argument("--allow-overwrite-snapshot", action="store_true")
    parser.add_argument("--debug-matching", action="store_true")
    parser.add_argument("--debug-geo-unmatched", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing API_KEY environment variable.")

    year = args.year
    month = args.month
    if year is None or month is None:
        year, month = last_complete_month(date.today())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = download_erni_pdbf_fase_i(api_key, year, month)
    if pdf_path is None:
        raise RuntimeError(f"PDF not found on ESIOS for {year:04d}-{month:02d}.")

    df_pdf = pdf_parser(year, month)

    mapping_path = DATA_DIR / "nodes_with_province_clean.xlsx"
    if not mapping_path.exists():
        raise FileNotFoundError(str(mapping_path))

    df_prov = province_aggregation_from_pdf_light(
        df_pdf,
        mapping_path,
        similarity_cutoff=float(args.similarity_cutoff),
        debug=bool(args.debug_matching),
        as_percent=True,
        percent_decimals=2,
    )

    json_path = OUT_DIR / "province_metrics.json"
    store_province_metrics_light_json(
        df_prov,
        out_path=json_path,
        year=year,
        month=month,
        mode="append",
        allow_overwrite_snapshot=bool(args.allow_overwrite_snapshot),
    )

    logo_path = DATA_DIR / "Logo.png"
    if not logo_path.exists():
        raise FileNotFoundError(str(logo_path))

    plot_two_province_heatmaps_from_json(
        json_path,
        year=year,
        month=month,
        logo_path=logo_path,
        out_dir=OUT_DIR,
        save_png=True,
        dpi=250,
        debug_unmatched=bool(args.debug_geo_unmatched),
    )

    print(f"Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()

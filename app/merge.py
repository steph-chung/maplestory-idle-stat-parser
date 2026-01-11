# app/merge.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_NUM_RE = re.compile(r"^\s*([0-9][0-9,]*)(?:\.([0-9]+))?\s*([%KM]?)\s*$", re.IGNORECASE)


def _parse_value(raw: str) -> Tuple[float, str, bool]:
    """
    Parse value strings like:
      - "23,378" -> (23378.0, "", True)
      - "97.8%"  -> (97.8, "%", True)
      - "24M852K" -> not handled here (returns (0,"",False)) because it's compound
      - "1.2K" -> (1200.0, "", True) (K/M treated as multipliers if single suffix)
    """
    s = (raw or "").strip().replace(" ", "")
    # Quick reject for compound cases like 24M852K
    if s.count("M") + s.count("K") > 1:
        return 0.0, "", False

    m = _NUM_RE.match(s)
    if not m:
        return 0.0, "", False

    int_part = m.group(1).replace(",", "")
    frac_part = m.group(2) or ""
    suffix = (m.group(3) or "").upper()

    num_str = int_part + (("." + frac_part) if frac_part else "")
    try:
        val = float(num_str)
    except ValueError:
        return 0.0, "", False

    if suffix == "K":
        val *= 1_000.0
        suffix = ""  # normalize unit away
    elif suffix == "M":
        val *= 1_000_000.0
        suffix = ""  # normalize unit away
    elif suffix == "%":
        # keep percent unit
        pass
    elif suffix == "":
        pass
    else:
        return 0.0, "", False

    return val, suffix, True


def merge_job_result(
    per_file_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge the Celery SUCCESS result (list of {filename, stats, ...}) into:
      - per_stat: merged value (prefers maximum numeric where comparable; otherwise most common)
      - coverage: how many files had the stat
      - per_file: pass-through for convenience

    Strategy per stat:
      1) If values parse as numeric with same unit (% vs non-%):
           merged_value = max numeric (common for "best" character snapshot)
      2) Else:
           merged_value = most common string (mode), tie-breaker: longest (more specific)

    Returns a dict suited for the API summary endpoint.
    """
    if not isinstance(per_file_results, list):
        raise ValueError("Expected a list of per-file results")

    # Collect values per stat
    values_by_stat: Dict[str, List[str]] = {}
    for item in per_file_results:
        stats = item.get("stats") or {}
        if not isinstance(stats, dict):
            continue
        for stat, val in stats.items():
            values_by_stat.setdefault(stat, []).append(str(val))

    merged: Dict[str, Any] = {}
    for stat, vals in values_by_stat.items():
        parsed = []
        ok_unit = None
        all_numeric = True

        for v in vals:
            num, unit, ok = _parse_value(v)
            if not ok:
                all_numeric = False
                break
            if ok_unit is None:
                ok_unit = unit
            elif unit != ok_unit:
                # unit mismatch (e.g. % vs none) -> treat as non-numeric comparable
                all_numeric = False
                break
            parsed.append((num, v))

        if all_numeric and parsed:
            # choose the raw string corresponding to the max numeric
            parsed.sort(key=lambda t: t[0], reverse=True)
            merged_value = parsed[0][1]
            merged[stat] = {
                "value": merged_value,
                "method": "max_numeric",
                "coverage": len(vals),
            }
        else:
            # mode / most common string
            freq: Dict[str, int] = {}
            for v in vals:
                freq[v] = freq.get(v, 0) + 1

            # sort by frequency desc, then length desc, then lexicographic
            ranked = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
            merged_value = ranked[0][0]
            merged[stat] = {
                "value": merged_value,
                "method": "mode_string",
                "coverage": len(vals),
            }

    summary = {
        "files": [
            {
                "filename": it.get("filename"),
                "stats": it.get("stats", {}),
                "error": it.get("error"),
            }
            for it in per_file_results
        ],
        "merged_stats": {k: v["value"] for k, v in merged.items()},
        "merged_detail": merged,  # includes method + coverage per stat
        "num_files": len(per_file_results),
        "num_stats_merged": len(merged),
    }
    return summary

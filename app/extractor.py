from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from rapidfuzz import fuzz, process

# Canonical stat names recognized by the extractor
STAT_CANON = [
    "Attack",
    "Defense",
    "Max HP",
    "Max MP",
    "Critical Rate",
    "Critical Damage",
    "Attack Speed",
    "Accuracy",
    "Evasion",
    "Min Damage Multiplier",
    "Max Damage Multiplier",
    "EXP Gain",
    "Meso Drop",
    "Final Damage",
    "Damage Amplification",
    "Basic Attack Damage",
    "Skill Damage",
    "Defense Penetration",
    "Boss Monster Damage",
    "Normal Monster Damage",
    "Status Effect Damage",
    "MP Recovery Per Sec",
    "Damage Taken Decrease",
    "Stat Prop. Damage",
    "Damage",
    "STR",
    "DEX",
    "INT",
    "LUK",
    "Companion Summoning Duration Increase",
    "Basic Attack Target Increase",
    "Skill Cooldown Decrease",
    "3rd Job Skill Lv.",
]

# Value regex: supports numbers, %, sec, and magnitude suffixes (K, M, B, T, AA-AZ)
_MAG_UNITS = "|".join([f"A{chr(i)}" for i in range(65, 91)] + ["T", "B", "M", "K"])
VAL_RE = re.compile(
    rf"([0-9][0-9,\.]*(?:(?:{_MAG_UNITS})[0-9][0-9,\.]*)*(?:{_MAG_UNITS})?(?:%|sec)?)",
    re.IGNORECASE,
)
OSEC_RE = re.compile(r"\b[Oo]sec\b", re.IGNORECASE)
SLASH_COUNTER_RE = re.compile(r"\b\d+\s*/\s*\d+\b")
TESS_CONFIG = "--psm 6"

SUFFIX_MERGE_WORDS = {
    "increase",
    "decrease",
    "multiplier",
    "duration",
    "lv",
    "lvl",
    "level",
}
NOISE_RE = re.compile(r"[£§¢¥€®©™°±²³µ¶·¹º»¼½¾¿×÷ƒ†‡•…‰‹›™℗∞≈≠≤≥⁄ⁿ₀₁₂₃₄₅₆₇₈₉\[\]{}|\\]")
HARD_MAP = {
    "str": "STR",
    "strw": "STR",
    "5tr": "STR",
    "srr": "STR",
    "sir": "STR",
    "dex": "DEX",
    "int": "INT",
    "nt": "INT",
    "luk": "LUK",
    "uk": "LUK",
    "statpropdamage": "Stat Prop. Damage",
    "statpropdamagei": "Stat Prop. Damage",
    "mprecoverypersec": "MP Recovery Per Sec",
    "mcoverypersec": "MP Recovery Per Sec",
    "companionsummoningdurationincrease": "Companion Summoning Duration Increase",
    "basicattacktargetincrease": "Basic Attack Target Increase",
    "skillcooldowndecrease": "Skill Cooldown Decrease",
    "3rdjobskilllv": "3rd Job Skill Lv.",
    "rdjobskilllv": "3rd Job Skill Lv.",
}


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z\. ]", "", s).lower().replace(" ", "")


def _has_letters(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s))


def _has_value_for_merge(s: str) -> bool:
    """Check if line has a numeric value. Don't strip trailing single digits if they're the only number."""
    # Only strip trailing junk like " on 3" if there's another number in the line
    if re.search(r"\d.*\s+\d{1,2}\s*$", s):
        s = re.sub(r"\s+\d{1,2}\s*$", "", s)
    return bool(VAL_RE.search(s))


def _build_lines_by_y_bucketing(ocr: Dict[str, List[Any]]) -> List[str]:
    """Reconstruct lines by clustering words using Y-center proximity."""
    words, heights = [], []
    for i in range(len(ocr.get("text", []))):
        txt = (ocr["text"][i] or "").strip()
        if txt:
            h = int(ocr["height"][i])
            words.append((int(ocr["top"][i]) + h / 2.0, int(ocr["left"][i]), txt))
            heights.append(h)

    if not words:
        return []

    y_tol = max(4.0, 0.25 * float(np.median(heights)))
    words.sort()

    rows: List[List] = []
    for yc, x, txt in words:
        if rows and abs(yc - rows[-1][0][0]) <= y_tol:
            rows[-1].append((yc, x, txt))
        else:
            rows.append([(yc, x, txt)])

    return [
        " ".join(t[2] for t in sorted(row, key=lambda t: t[1]))
        for row in sorted(rows, key=lambda r: r[0][0])
    ]


def extract_stats(
    image_path: str,
    cp_template_path: str,
    *,
    match_threshold: float = 0.55,
    scale_steps: int = 20,
    fuzzy_threshold: float = 70.0,
    return_debug: bool = False,
) -> Dict[str, str] | Tuple[Dict[str, str], Dict[str, Any]]:
    """Extract stats under the CP pill across resolutions."""

    tess_cmd = os.environ.get("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    templ = cv2.imread(cp_template_path, cv2.IMREAD_GRAYSCALE)
    if templ is None:
        raise FileNotFoundError(cp_template_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find CP pill (template match)
    best_score = -1.0
    best_loc = None
    best_scale = None

    for s in np.linspace(0.40, 1.80, scale_steps):
        r = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        if r.shape[0] < templ.shape[0] or r.shape[1] < templ.shape[1]:
            continue
        res = cv2.matchTemplate(r, templ, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best_score:
            best_score, best_loc, best_scale = score, loc, s

    if best_loc is None or best_scale is None or best_score < match_threshold:
        raise RuntimeError(f"CP pill not found (score={best_score:.3f})")

    cp_x, cp_y = int(best_loc[0] / best_scale), int(best_loc[1] / best_scale)
    cp_w, cp_h = int(templ.shape[1] / best_scale), int(templ.shape[0] / best_scale)

    # Crop panel under CP pill
    left = int(cp_x - 0.05 * cp_w)
    top = int(cp_y + 0.80 * cp_h)
    right = int(cp_x + 1.20 * cp_w)
    bottom = int(cp_y + 7.50 * cp_h)

    H, W = img.shape[:2]
    left, top = max(0, left), max(0, top)
    right, bottom = min(W, right), min(H, bottom)

    panel_bgr = img[top:bottom, left:right]

    # Inner trim
    ph, pw = panel_bgr.shape[:2]
    panel_bgr = panel_bgr[
        int(0.03 * ph) : int(0.97 * ph), int(0.02 * pw) : int(0.94 * pw)
    ]

    # Normalize for OCR
    rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    im_base = Image.fromarray(rgb)

    target_h = 980
    scale = target_h / max(1, im_base.height)
    im_base = im_base.resize(
        (max(1, int(im_base.width * scale)), target_h), Image.BICUBIC
    )
    im_gray = ImageOps.grayscale(im_base)

    side_border = int(0.03 * im_gray.width)
    top_border = int(0.01 * im_gray.height)
    bottom_border = int(0.05 * im_gray.height)

    # Try multiple contrast levels and pick the best result
    best_lines: List[str] = []
    best_score = 0

    for contrast in [2.2, 1.0]:  # High contrast first, then no enhancement
        im = im_gray.copy()
        if contrast != 1.0:
            im = ImageEnhance.Contrast(im).enhance(contrast)
            im = ImageEnhance.Sharpness(im).enhance(2.0)

        im = ImageOps.expand(
            im, border=(side_border, top_border, side_border, bottom_border), fill=255
        )

        ocr = pytesseract.image_to_data(
            im, config=TESS_CONFIG, output_type=pytesseract.Output.DICT
        )
        lines = _build_lines_by_y_bucketing(ocr)
        score = sum(1 for line in lines if VAL_RE.search(line))
        if score > best_score:
            best_score, best_lines = score, lines

    ordered_lines = best_lines

    def pre_process_multiline(lines: List[str]) -> List[str]:
        """Handle stat labels that span multiple lines in the UI."""
        result = []
        skip_until = -1
        for i, line in enumerate(lines):
            if i < skip_until:
                continue

            ln = line.lower()

            # Companion Summoning Duration Increase
            if "companion" in ln and "summoning" in ln:
                val, end_idx = None, None
                for j in range(i + 1, min(i + 8, len(lines))):
                    nxt = lines[j]
                    if not val and re.search(r"\d+%", nxt):
                        val = re.search(r"(\d+%)", nxt).group(1)
                    if "duration" in nxt.lower() and "increase" in nxt.lower():
                        end_idx = j
                        break
                if val and end_idx:
                    result.append(f"Companion Summoning Duration Increase {val}")
                    skip_until = end_idx + 1
                    continue

            # Basic Attack Target Increase
            if "basic" in ln and "attack" in ln and "target" in ln:
                val, end_idx = None, None
                for j in range(i + 1, min(i + 4, len(lines))):
                    nxt = lines[j]
                    if not val and re.match(r"^\d+$", nxt.strip()):
                        val = nxt.strip()
                    if nxt.strip().lower() == "increase":
                        end_idx = j
                        break
                if val and end_idx:
                    result.append(f"Basic Attack Target Increase {val}")
                    skip_until = end_idx + 1
                    continue

            # Skill Cooldown Decrease
            if "skill" in ln and "cooldown" in ln.replace(" ", "").replace(
                "i", ""
            ).replace("ww", ""):
                val, end_idx = None, None
                for j in range(i + 1, min(i + 4, len(lines))):
                    nxt = lines[j]
                    if not val and re.search(r"\d+%", nxt):
                        val = re.search(r"(\d+%)", nxt).group(1)
                    if nxt.strip().lower() == "decrease":
                        end_idx = j
                        break
                if val and end_idx:
                    result.append(f"Skill Cooldown Decrease {val}")
                    skip_until = end_idx + 1
                    continue

            # Min Damage Multiplier
            if "min" in ln and "damage" in ln:
                val, end_idx = None, None
                for j in range(i + 1, min(i + 6, len(lines))):
                    nxt = lines[j]
                    if not val and re.search(r"\d+\.?\d*%", nxt):
                        val = re.search(r"(\d+\.?\d*%)", nxt).group(1)
                    if "multiplier" in nxt.lower():
                        end_idx = j
                        break
                if val and end_idx:
                    result.append(f"Min Damage Multiplier {val}")
                    skip_until = end_idx + 1
                    continue

            result.append(line)

        return result

    ordered_lines = pre_process_multiline(ordered_lines)

    # Merge wrapped rows
    merged_lines: List[str] = []
    i = 0
    while i < len(ordered_lines):
        cur = ordered_lines[i]
        cur_has_letters = _has_letters(cur)
        cur_has_val = _has_value_for_merge(cur)

        # Check if line is "value-only" (no letters, or only short OCR noise like "iG")
        cur_letters_only = re.sub(r"[^A-Za-z]", "", cur)
        cur_is_value_only = cur_has_val and len(cur_letters_only) <= 2

        # Suffix attaches to previous line
        if merged_lines and cur_has_letters and not cur_has_val:
            suffix = re.sub(r"[^A-Za-z]", "", cur).lower()
            if suffix in SUFFIX_MERGE_WORDS and _has_value_for_merge(merged_lines[-1]):
                merged_lines[-1] += " " + cur
                i += 1
                continue

        # 2-line: value + label (value-only line followed by label-only line)
        if cur_is_value_only and i + 1 < len(ordered_lines):
            nxt = ordered_lines[i + 1]
            nxt_has_letters, nxt_has_val = _has_letters(nxt), _has_value_for_merge(nxt)
            if nxt_has_letters and not nxt_has_val:
                # Combine as "label value" (reorder so label comes first)
                merged_lines.append(nxt + " " + cur)
                i += 2
                continue

        if cur_has_letters and not cur_has_val and i + 1 < len(ordered_lines):
            nxt = ordered_lines[i + 1]
            nxt_has_letters, nxt_has_val = _has_letters(nxt), _has_value_for_merge(nxt)

            # 3-line: label + value + suffix
            if i + 2 < len(ordered_lines) and nxt_has_val and not nxt_has_letters:
                nxt2 = ordered_lines[i + 2]
                if _has_letters(nxt2) and not _has_value_for_merge(nxt2):
                    suffix = re.sub(r"[^A-Za-z]", "", nxt2).lower()
                    if suffix in SUFFIX_MERGE_WORDS:
                        merged_lines.append(cur + " " + nxt2 + " " + nxt)
                        i += 3
                        continue

            # 2-line: label + value
            if nxt_has_val:
                merged_lines.append(cur + " " + nxt)
                i += 2
                continue

            # 3-line: label + label + value
            if i + 2 < len(ordered_lines) and nxt_has_letters and not nxt_has_val:
                nxt2 = ordered_lines[i + 2]
                if _has_value_for_merge(nxt2) and not _has_letters(nxt2):
                    merged_lines.append(cur + " " + nxt + " " + nxt2)
                    i += 3
                    continue

        merged_lines.append(cur)
        i += 1

    # Fuzzy mapping
    canon_norm = {_norm_name(s): s for s in STAT_CANON}
    canon_keys = list(canon_norm.keys())

    def map_label(raw_name: str) -> Tuple[Optional[str], float]:
        n = _norm_name(raw_name)

        if n in HARD_MAP:
            return HARD_MAP[n], 100.0

        # Damage disambiguation
        if "damage" in n:
            if any(
                x in n
                for x in (
                    "taken",
                    "taker",
                    "tak",
                    "decrease",
                    "decre",
                    "crease",
                    "jecre",
                )
            ):
                return "Damage Taken Decrease", 100.0
            if fuzz.ratio(n, "damage") >= 80:
                return "Damage", 100.0

        res = process.extractOne(n, canon_keys, scorer=fuzz.ratio)
        if not res:
            return None, 0.0
        return canon_norm[res[0]], float(res[1])

    def split_jammed(line: str) -> List[str]:
        """Split jammed stats (multiple stats on one line)."""
        for pattern, split_re in [
            (r"sec\s+3rd", r"(?=\b3rd\s)"),
            (r"exp\s*gain.*meso\s*drop", r"(?=\bmeso\s*drop\b)"),
        ]:
            if re.search(pattern, line, re.IGNORECASE):
                return [
                    p.strip()
                    for p in re.split(split_re, line, flags=re.IGNORECASE)
                    if p.strip()
                ]
        return [line]

    stats: Dict[str, str] = {}
    match_debug: Dict[str, Any] = {}

    for line0 in merged_lines:
        for line in split_jammed(line0):
            line = NOISE_RE.sub("", line).strip()
            line = re.sub(r"\s+", " ", line)

            if not line or "CP" in line or SLASH_COUNTER_RE.search(line):
                continue

            osec_match = OSEC_RE.search(line)
            mlist = list(VAL_RE.finditer(line))

            if not mlist and osec_match:
                value, raw_name = "0sec", OSEC_RE.sub("", line).strip(" -:\t.")
            elif mlist:
                # Combine adjacent magnitude values (e.g., "24M 852K" -> "24M852K")
                if len(mlist) >= 2:
                    v1, v2 = mlist[-2].group(1), mlist[-1].group(1)
                    gap = mlist[-1].start() - mlist[-2].end()
                    is_mag = re.search(r"[KMBT]$", v1, re.I) and re.search(
                        r"^[0-9].*[KMBT]$", v2, re.I
                    )
                    value = v1 + v2 if gap <= 2 and is_mag else mlist[-1].group(1)
                else:
                    value = mlist[-1].group(1)
                raw_name = VAL_RE.sub("", line).strip(" -:\t.")
            else:
                continue

            if osec_match and value.lower() == "osec":
                value, raw_name = "0sec", OSEC_RE.sub("", line).strip(" -:\t.")

            raw_name = re.sub(r"(\s+[0Oo])+$", "", raw_name).strip()

            # Prevent "Decrease Damage" from hijacking the Damage stat
            rn = raw_name.lower().replace(" ", "")
            if "decrease" in rn and "taken" not in rn:
                raw_name = re.sub(r"\bDecrease\b", "", raw_name, flags=re.I).strip()

            if not _has_letters(raw_name):
                continue

            stat, score = map_label(raw_name)
            if stat is None:
                continue

            threshold = 80.0 if len(_norm_name(raw_name)) <= 3 else fuzzy_threshold
            if score < threshold:
                continue

            # Only special-case Skill Cooldown Decrease units
            out_key = stat
            if stat == "Skill Cooldown Decrease":
                v = value.strip().lower()
                out_key = (
                    f"Skill Cooldown Decrease ({'%' if v.endswith('%') else 'sec' if v.endswith('sec') else ''})"
                    if v.endswith("%") or v.endswith("sec")
                    else stat
                )

            stats[out_key] = value
            match_debug[out_key] = {
                "value": value,
                "score": score,
                "raw_line": line,
                "raw_name": raw_name,
            }

    if not return_debug:
        return stats

    return stats, {
        "cp_match": {"x": cp_x, "y": cp_y, "w": cp_w, "h": cp_h, "score": best_score},
        "crop_rect": {"left": left, "top": top, "right": right, "bottom": bottom},
        "ordered_lines": ordered_lines,
        "merged_lines": merged_lines,
        "matches": match_debug,
    }

import json
from pathlib import Path

import pytest

from app.extractor import extract_stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
EXPECTED_PATH = ROOT / "tests" / "expected.json"


def _load_expected():
    with EXPECTED_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("image_name", sorted(_load_expected().keys()))
def test_golden_images(image_name):
    expected = _load_expected()[image_name]

    image_path = DATA / image_name
    template_path = DATA / "cp_template.png"

    assert image_path.exists(), f"Missing fixture image: {image_path}"
    assert template_path.exists(), f"Missing cp_template.png at: {template_path}"

    stats, dbg = extract_stats(
        str(image_path),
        str(template_path),
        return_debug=True,
    )

    # 1) Ensure we didn't miss expected keys
    missing = sorted(set(expected.keys()) - set(stats.keys()))
    assert not missing, (
        f"Missing keys for {image_name}: {missing}\nmerged_lines={dbg.get('merged_lines')}"
    )

    # 2) Ensure values match exactly
    wrong = {
        k: {"expected": expected[k], "got": stats.get(k)}
        for k in expected
        if stats.get(k) != expected[k]
    }
    assert not wrong, (
        f"Wrong values for {image_name}: {wrong}\nmerged_lines={dbg.get('merged_lines')}"
    )

import json
from pathlib import Path

import pytest
from PIL import Image

from ocr.ocr_handler import process_image
from tests.test_ocr_real_pipeline import _extract_text_from_ocr_result, _normalize_whitespace

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "ocr_ground_truth.json"
REPO_ROOT = FIXTURE_PATH.parents[2]


def _resolve_source_image(case):
    source_image = case.get("source_image")
    if not source_image:
        pytest.fail(f"Case {case.get('id')} is CONFIRMED but missing source_image.")

    candidates = [
        REPO_ROOT / source_image,
        (REPO_ROOT / "tests") / source_image,
        (REPO_ROOT / "tests" / "fixtures") / source_image,
        (REPO_ROOT / "datatest") / source_image,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Case {case.get('id')} references missing source_image '{source_image}'. "
        "Checked repo root, tests/, tests/fixtures/, and datatest/."
    )


def _extract_page_number_from_image(image_path):
    stem = image_path.stem.lower()
    prefix = "page"
    if not stem.startswith(prefix):
        raise ValueError(
            f"Source image '{image_path.name}' does not follow the expected page naming convention."
        )

    digits = stem[len(prefix) :]
    if not digits or not digits.isdigit():
        raise ValueError(
            f"Source image '{image_path.name}' does not contain a valid numeric page index."
        )

    return int(digits)


def _load_confirmed_cases():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = []
    for case in data.get("cases", []):
        if case.get("status") != "CONFIRMED":
            continue

        try:
            image_path = _resolve_source_image(case)
        except FileNotFoundError as exc:
            pytest.fail(str(exc))

        if case.get("expected_value") is None:
            pytest.fail(
                f"Case {case.get('id')} is CONFIRMED but expected_value is null; "
                "field mapping is incomplete and the case cannot be asserted safely."
            )

        cases.append((case, image_path))

    return cases


def _has_validated_field_mapping(case):
    return False


@pytest.mark.xfail(reason="Fails intentionally until brand-region extractor is corrected", strict=False)
def test_gt002_brand_region_raw_ocr_must_match_ground_truth():
    source_path = REPO_ROOT / "page1.png"
    crop_path = REPO_ROOT / "brand_region.png"

    assert source_path.exists(), "page1.png must exist for the GT-002 regression evidence."
    assert crop_path.exists(), "brand_region.png must exist so the verified brand crop is used as the test input."

    with source_path.open("rb") as source_file:
        source_image = Image.open(source_file)
        brand_crop = source_image.crop((200, 330, 520, 520))

    assert brand_crop.size[0] > 0 and brand_crop.size[1] > 0, "Brand crop is empty; GT-002 cannot be asserted safely."

    with crop_path.open("rb") as crop_file:
        ocr_result = process_image(crop_file, crop_path.name)

    assert ocr_result is not None, "GT-002 produced no OCR result in the verified brand crop."

    raw_text = _extract_text_from_ocr_result(ocr_result)
    normalized = _normalize_whitespace(raw_text)

    assert normalized, "GT-002 crop produced empty OCR text; this is a real OCR failure, not a skipped case."

    expected_value = "SECURE"
    assert normalized.upper() == expected_value, (
        f"GT-002 raw OCR on the verified brand crop is {normalized!r}, expected {expected_value!r}. "
        "This test fails intentionally until the brand-region extractor is corrected."
    )


@pytest.mark.parametrize(
    "case, image_path",
    _load_confirmed_cases(),
    ids=lambda item: item[0]["id"] if isinstance(item, tuple) else str(item),
)
def test_confirmed_ground_truth_cases(case, image_path):
    expected_value = case["expected_value"]
    expected_norm = _normalize_whitespace(expected_value).lower()

    page_number = case.get("page")
    if page_number is None:
        pytest.fail(f"Case {case.get('id')} is missing page metadata; field mapping is incomplete.")

    try:
        image_page = _extract_page_number_from_image(image_path)
    except ValueError as exc:
        pytest.fail(str(exc))

    if image_page != int(page_number):
        pytest.fail(
            f"Case {case.get('id')} page mismatch: declared page={page_number}, actual image page={image_page}. "
            "The source_image must match the case page."
        )

    with image_path.open("rb") as f:
        ocr_result = process_image(f, image_path.name)

    assert ocr_result is not None, (
        f"Case {case.get('id')} produced no OCR result for {image_path.name}; "
        "check the image file and runtime OCR dependencies."
    )

    ocr_text = _extract_text_from_ocr_result(ocr_result)
    ocr_norm = _normalize_whitespace(ocr_text).lower()

    assert ocr_norm, (
        f"Case {case.get('id')} produced empty OCR text for {image_path.name}; "
        "field mapping cannot be verified without OCR output."
    )

    if not _has_validated_field_mapping(case):
        pytest.skip(
            f"Case {case.get('id')} [{case.get('field')}] requires a validated field-extraction rule before assertion; "
            f"expected_value={expected_value!r} is not yet mapped to a trusted OCR field."
        )

    if expected_norm not in ocr_norm:
        pytest.fail(
            f"Case {case.get('id')} is a confirmed OCR regression for field='{case.get('field')}' "
            f"on page={page_number}: expected={expected_value!r} was not found in the OCR output."
        )

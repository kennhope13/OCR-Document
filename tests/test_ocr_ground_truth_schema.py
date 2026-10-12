import json
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ocr_ground_truth.json"
ALLOWED_STATUSES = {"CONFIRMED", "NOT_AN_ERROR", "PENDING"}
CASE_REQUIRED_KEYS = {
    "id",
    "page",
    "field",
    "source_image",
    "expected_value",
    "actual_ocr",
    "status",
    "notes",
}
PENDING_REQUIRED_KEYS = {"id", "page", "field", "source_image", "status", "notes"}


def _load_fixture():
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_fixture_is_valid_json_and_has_expected_top_level_structure():
    payload = _load_fixture()

    assert isinstance(payload, dict)
    assert payload["document"] == "DS-K1107AM.pdf"
    assert isinstance(payload["cases"], list)
    assert isinstance(payload["pending_cases"], list)


def test_cases_have_required_keys_and_expected_types():
    payload = _load_fixture()

    for case in payload["cases"]:
        missing = CASE_REQUIRED_KEYS - set(case.keys())
        assert not missing, f"Case {case.get('id')} missing keys: {sorted(missing)}"

        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["page"], int)
        assert isinstance(case["field"], str) and case["field"]
        assert isinstance(case["source_image"], str) and case["source_image"]
        assert isinstance(case["status"], str)
        assert isinstance(case["notes"], str)
        assert case["status"] in ALLOWED_STATUSES

        if case["status"] == "CONFIRMED":
            assert isinstance(case["expected_value"], str)
            assert isinstance(case["actual_ocr"], str)

        if case["status"] == "NOT_AN_ERROR":
            assert isinstance(case["expected_value"], str)
            assert isinstance(case["actual_ocr"], str)


def test_pending_cases_have_required_tracking_fields_and_are_not_mixed_into_confirmed_cases():
    payload = _load_fixture()

    pending_ids = []
    for item in payload["pending_cases"]:
        missing = PENDING_REQUIRED_KEYS - set(item.keys())
        assert not missing, f"Pending case {item.get('id')} missing keys: {sorted(missing)}"
        assert item["status"] == "PENDING"
        assert item["status"] in ALLOWED_STATUSES
        pending_ids.append(item["id"])

    case_ids = [case["id"] for case in payload["cases"]]
    for pending_id in pending_ids:
        assert pending_id not in case_ids

    for case in payload["cases"]:
        assert case["status"] != "PENDING", (
            f"Case {case.get('id')} is marked as PENDING in the confirmed list; "
            "pending records must remain in pending_cases."
        )


def test_ids_are_unique_across_confirmed_and_pending_sets():
    payload = _load_fixture()

    ids = [case["id"] for case in payload["cases"]]
    ids.extend(item["id"] for item in payload["pending_cases"])
    assert len(ids) == len(set(ids)), "Fixture IDs must be unique across confirmed and pending cases."


def test_status_counts_match_expected_fixture_state():
    payload = _load_fixture()

    confirmed = sum(1 for case in payload["cases"] if case["status"] == "CONFIRMED")
    not_an_error = sum(1 for case in payload["cases"] if case["status"] == "NOT_AN_ERROR")
    pending = len(payload["pending_cases"])

    assert confirmed == 10, f"Expected 10 CONFIRMED cases, found {confirmed}."
    assert not_an_error == 1, f"Expected 1 NOT_AN_ERROR case, found {not_an_error}."
    assert pending == 2, f"Expected 2 PENDING cases, found {pending}."


def test_valid_status_values_are_used_only_for_real_ocr_outcomes():
    payload = _load_fixture()

    confirmed_ids = [case["id"] for case in payload["cases"] if case["status"] == "CONFIRMED"]
    not_an_error_ids = [case["id"] for case in payload["cases"] if case["status"] == "NOT_AN_ERROR"]

    assert confirmed_ids, "At least one confirmed OCR case is required."
    assert not_an_error_ids, "At least one non-error OCR case is required."

    for case in payload["cases"]:
        assert case["status"] in ALLOWED_STATUSES

    for item in payload["pending_cases"]:
        assert item["status"] == "PENDING"

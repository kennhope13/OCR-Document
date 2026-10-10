import pytest
from ie.extractor import DocumentIE

def test_extract_date_success():
    ie = DocumentIE()
    blocks = [
        "We,Hangzhou Hikvision Technology Co.Ltd,certify on May 28, 2024 the products called",
        "for by Sales Contract No.HIKNHA0524-03"
    ]
    result = ie.extract_date(blocks)
    assert result.status == "candidate"
    assert result.value == "May 28, 2024"
    assert result.method == "regex_context"

def test_extract_date_with_punctuation():
    ie = DocumentIE()
    blocks = ["dated: May 28, 2024."]
    result = ie.extract_date(blocks)
    assert result.status == "candidate"
    assert result.value == "May 28, 2024"

def test_extract_date_not_found():
    ie = DocumentIE()
    blocks = ["We certify the products on this document."]
    result = ie.extract_date(blocks)
    assert result.status == "failed"
    assert result.value == ""

def test_extract_date_invalid_calendar_date():
    ie = DocumentIE()
    blocks = ["certify on February 30, 2024"]
    result = ie.extract_date(blocks)
    # Vì February 30 không hợp lệ nên sẽ bị bỏ qua -> failed
    assert result.status == "failed"
    assert result.value == ""

def test_extract_date_multiple_dates():
    ie = DocumentIE()
    blocks = [
        "This document was printed on January 1, 2023.", # Random date without target context
        "We certify on May 28, 2024 that the products are genuine.", # Expected contextual date
        "Valid until December 31, 2025." # Another random date
    ]
    result = ie.extract_date(blocks)
    assert result.status == "candidate"
    # Dù January 1, 2023 xuất hiện trước, hệ thống phải ưu tiên cụm đi kèm "certify on"
    assert result.value == "May 28, 2024"

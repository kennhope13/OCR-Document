import pytest
from ie.extractor import DocumentIE

def test_extract_recipient_single_line():
    ie = DocumentIE()
    blocks = [
        "SOME OTHER TEXT",
        "SECURE HOUSE JOINT STOCK COMPANY (NHA AN TOAN.JSC)",
        "5TH FLOOR, HANOI",
        "We, Hangzhou Hikvision Technology Co.Ltd, certify"
    ]
    result = ie.extract_recipient_company(blocks)
    assert result.status == "candidate"
    assert result.value == "SECURE HOUSE JOINT STOCK COMPANY (NHA AN TOAN.JSC)"

def test_extract_recipient_multiline():
    ie = DocumentIE()
    blocks = [
        "SEEURE",
        "HOUSE JOINT STOCK COMPANY (NHA AN TOAN.JSC)",
        "5TH FLOOR, 97-99 LANG HA BUILDING, HANOI",
        "We, Hangzhou Hikvision Technology Co.Ltd, certify"
    ]
    result = ie.extract_recipient_company(blocks)
    assert result.status == "candidate"
    assert result.value == "SEEURE HOUSE JOINT STOCK COMPANY (NHA AN TOAN.JSC)"

def test_extract_recipient_stop_at_address_and_vn():
    ie = DocumentIE()
    blocks = [
        "CONG TY",
        "CO PHAN",
        "NHA AN TOAN",
        "SEEURE",
        "HOUSE JOINT STOCK COMPANY (NHA AN TOAN.JSC)",
        "5TH FLOOR, 97-99 LANG HA BUILDING, LANG HA WARD",
        "DISTRICT, HANOI, VIETNAM",
        "We, Hangzhou Hikvision Technology Co.Ltd, certify"
    ]
    result = ie.extract_recipient_company(blocks)
    assert result.status == "candidate"
    # Should not include Vietnamese part or address part
    assert result.value == "SEEURE HOUSE JOINT STOCK COMPANY (NHA AN TOAN.JSC)"

def test_extract_recipient_issuer_not_selected():
    ie = DocumentIE()
    blocks = [
        "CERTIFICATE OF QUALITY",
        "We, Hangzhou Hikvision Technology Co.Ltd, certify"
    ]
    result = ie.extract_recipient_company(blocks)
    assert result.status == "failed"
    assert result.value == ""

def test_extract_recipient_not_found():
    ie = DocumentIE()
    blocks = [
        "Just a random document without header company",
        "We, certify that something is true."
    ]
    result = ie.extract_recipient_company(blocks)
    assert result.status == "failed"
    assert result.value == ""

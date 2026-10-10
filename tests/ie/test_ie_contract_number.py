import pytest
from ie.extractor import DocumentIE

def test_extract_contract_success():
    ie = DocumentIE()
    blocks = [
        "for by Sales Contract No.HIKNHA0524-03 CI No.A24011321062) are furnished in"
    ]
    result = ie.extract_contract_number(blocks)
    assert result.status == "candidate"
    assert result.value == "HIKNHA0524-03"

def test_extract_contract_split_blocks():
    ie = DocumentIE()
    blocks = [
        "for by Sales Contract No.",
        "HIKNHA0524-03",
        "CI No.A24011321062) are furnished in"
    ]
    result = ie.extract_contract_number(blocks)
    assert result.status == "candidate"
    assert result.value == "HIKNHA0524-03"

def test_extract_contract_with_punctuation():
    ie = DocumentIE()
    blocks = ["Sales Contract No: AB-1234."]
    result = ie.extract_contract_number(blocks)
    assert result.status == "candidate"
    assert result.value == "AB-1234"

def test_extract_contract_not_found():
    ie = DocumentIE()
    blocks = ["We certify the products on this document."]
    result = ie.extract_contract_number(blocks)
    assert result.status == "failed"
    assert result.value == ""

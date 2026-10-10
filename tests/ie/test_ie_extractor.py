import pytest
from ie.extractor import DocumentIE

def test_extract_contract_number_success():
    ie = DocumentIE()
    
    # Mô phỏng các line text được trả về từ OCR (Reconstructed Text)
    blocks = [
        "We,Hangzhou Hikvision Technology Co.Ltd,certify on May 28,2024 the products called",
        "for by Sales Contract No.HIKNHA0524-03 CI No.A24011321062) are furnished in",
        "accordance with all applicable requirements."
    ]
    
    result = ie.extract_contract_number(blocks)
    
    assert result.status == "candidate"
    assert result.value == "HIKNHA0524-03"
    assert result.method == "regex_context"
    # source_text phải lưu giữ lại đoạn văn bản chứa giá trị để đối chiếu
    assert result.source_text == "Sales Contract No.HIKNHA0524-03"

def test_extract_contract_number_not_found():
    ie = DocumentIE()
    blocks = [
        "This document does not contain any contract info.",
        "Just a normal sentence."
    ]
    
    result = ie.extract_contract_number(blocks)
    
    assert result.status == "failed"
    assert result.value == ""
    assert result.confidence == 0.0

def test_extract_contract_number_with_variations():
    ie = DocumentIE()
    # Test các định dạng OCR khác nhau
    blocks_1 = ["Contract No: ABC-12345 is valid"]
    blocks_2 = ["Sales contract no ABC-12345"] # missing dot
    
    assert ie.extract_contract_number(blocks_1).value == "ABC-12345"
    assert ie.extract_contract_number(blocks_2).value == "ABC-12345"

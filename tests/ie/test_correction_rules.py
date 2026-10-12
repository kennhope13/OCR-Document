import pytest
from ie.correction.rules import DateNormalizationRule, ContractNumberRule, CompanyAliasRule

def test_date_normalization_rule():
    rule = DateNormalizationRule()
    
    # Needs fix
    applied, val, reason = rule.apply("May 28,2024")
    assert applied is True
    assert val == "May 28, 2024"
    
    # Already fine
    applied, val, reason = rule.apply("May 28, 2024")
    assert applied is False
    assert val == "May 28, 2024"

def test_contract_number_rule():
    rule = ContractNumberRule()
    ctx = {"valid_contracts": ["HKNHNA0524-03", "SYN-NOI-2024"]}
    
    # Exact match - no correction
    applied, val, reason = rule.apply("HKNHNA0524-03", ctx)
    assert applied is False
    assert val == "HKNHNA0524-03"
    
    # OCR typo l vs I
    applied, val, reason = rule.apply("SYN-NOl-2024", ctx)
    assert applied is True
    assert val == "SYN-NOI-2024"
    
    # Unverified typo - no correction
    applied, val, reason = rule.apply("UNKNOWN-123", ctx)
    assert applied is False
    assert val == "UNKNOWN-123"

def test_company_alias_rule():
    rule = CompanyAliasRule()
    ctx = {
        "company_aliases": {
            "THE SECURE HOUSE JOINT STOCK COMPANY (NHA AN TOAN.,JSC)": [
                "SEEURE HOUSE", "HOUSE JOINT STOCK COMPANY"
            ],
            "BLURRY VISION JOINT STOCK COMPANY": [
                "BLURRYVISION"
            ]
        }
    }
    
    # Matches alias "SEEURE HOUSE"
    applied, val, reason = rule.apply("SEEURE HOUSE JOINT STOCK COMPANY", ctx)
    assert applied is True
    assert val == "THE SECURE HOUSE JOINT STOCK COMPANY (NHA AN TOAN.,JSC)"
    
    # Matches alias "BLURRYVISION" (missing space)
    applied, val, reason = rule.apply("BLURRYVISIONJOINT STOCK COMPANY", ctx)
    assert applied is True
    assert val == "BLURRY VISION JOINT STOCK COMPANY"
    
    # Unknown alias
    applied, val, reason = rule.apply("UNKNOWN CORP", ctx)
    assert applied is False
    assert val == "UNKNOWN CORP"

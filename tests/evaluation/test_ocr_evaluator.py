import pytest
from tests.evaluation.ocr_evaluator import (
    normalize_text,
    calculate_edit_operations,
    evaluate_text_pair
)

def test_normalize_text():
    assert normalize_text("  Hello   World  ") == "hello world"
    assert normalize_text("SECURE") == "secure"
    assert normalize_text(None) == ""

def test_calculate_edit_operations_string():
    # Substitute 1 char: SECURE -> SEEURE
    edits = calculate_edit_operations("secure", "seeure")
    assert edits["distance"] == 1
    assert edits["substitutions"] == 1
    assert edits["insertions"] == 0
    assert edits["deletions"] == 0
    
    # Deletion: adapter -> adaptr
    edits = calculate_edit_operations("adapter", "adaptr")
    assert edits["distance"] == 1
    assert edits["deletions"] == 1
    
    # Insertion: 50pcs -> 500pcs
    edits = calculate_edit_operations("50pcs", "500pcs")
    assert edits["distance"] == 1
    assert edits["insertions"] == 1
    
    # Complex
    edits = calculate_edit_operations("kitten", "sitting")
    # k->s (sub), e->i (sub), +g (ins)
    assert edits["distance"] == 3
    assert edits["substitutions"] == 2
    assert edits["insertions"] == 1

def test_calculate_edit_operations_words():
    edits = calculate_edit_operations(["hello", "world"], ["hello", "there", "world"])
    assert edits["distance"] == 1
    assert edits["insertions"] == 1

def test_evaluate_text_pair():
    result = evaluate_text_pair("60 PCS", "50PCS")
    assert result["expected_norm"] == "60 pcs"
    assert result["actual_norm"] == "50pcs"
    
    assert result["cer_edits"]["distance"] == 2
    assert result["cer_edits"]["ref_length"] == 6
    assert result["cer"] == 2 / 6
    
    # ["60", "pcs"] vs ["50pcs"] -> 1 sub, 1 del -> dist 2
    assert result["wer_edits"]["distance"] == 2
    assert result["wer_edits"]["ref_length"] == 2
    assert result["wer"] == 2 / 2
    
    assert result["strict_match"] is False
    assert result["normalized_match"] is False

import pytest
from ie.schema import ExtractedField
from ie.correction.correction_layer import CorrectionLayer

def test_correction_layer_immutability():
    layer = CorrectionLayer()
    original_field = ExtractedField(
        value="May 28,2024",
        source_text="May 28,2024",
        method="regex",
        status="candidate",
        confidence=0.9
    )
    
    # Store original values to check mutation
    orig_val = original_field.value
    
    new_field = layer.correct_field("date", original_field, {})
    
    # Check original wasn't mutated
    assert original_field.value == "May 28,2024"
    assert not hasattr(original_field, "corrected_value") or original_field.corrected_value is None
    
    # Check new field has correction
    assert new_field.value == "May 28, 2024"
    assert new_field.original_value == "May 28,2024"
    assert new_field.corrected_value == "May 28, 2024"
    assert new_field.correction_applied is True
    
    # Test double correction (should not change)
    new_field_2 = layer.correct_field("date", new_field, {})
    assert new_field_2.value == "May 28, 2024"
    assert new_field_2.correction_applied is False

def test_correction_layer_skips_failed():
    layer = CorrectionLayer()
    original_field = ExtractedField(
        value="",
        source_text="",
        method="regex",
        status="failed",
        confidence=0.0
    )
    
    new_field = layer.correct_field("date", original_field, {})
    assert new_field.value == ""
    assert new_field.correction_applied is False

def test_correction_layer_document():
    layer = CorrectionLayer()
    doc = {
        "date": ExtractedField(value="May 28,2024", source_text="May 28,2024", method="x", status="candidate"),
        "unknown_field": ExtractedField(value="XYZ", source_text="XYZ", method="x", status="candidate")
    }
    
    res = layer.correct_document(doc, {})
    
    assert res["date"].value == "May 28, 2024"
    assert res["date"].correction_applied is True
    
    assert res["unknown_field"].value == "XYZ"
    assert res["unknown_field"].correction_applied is False

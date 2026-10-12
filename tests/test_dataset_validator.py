import json
import pytest
from pathlib import Path
from data.ie_dataset_v2.validate_dataset import validate_dataset

@pytest.fixture
def temp_dataset(tmp_path):
    raw_dir = tmp_path / "raw"
    ann_dir = tmp_path / "annotations"
    raw_dir.mkdir()
    ann_dir.mkdir()
    
    # Create fake Image
    (raw_dir / "test.png").touch()
    
    yield tmp_path

def test_validator_success(temp_dataset):
    ann_dir = temp_dataset / "annotations"
    valid_json = {
        "document_id": "DOC-1",
        "source_file": "test.png",
        "fields": {
            "date": {
                "presence": "PRESENT",
                "verification_status": "CONFIRMED",
                "page": 1,
                "bounding_box": [10, 10, 50, 50]
            }
        }
    }
    with open(ann_dir / "valid.json", "w") as f:
        json.dump(valid_json, f)
        
    errors = validate_dataset(str(temp_dataset))
    assert len(errors) == 0

def test_validator_negative_cases(temp_dataset):
    ann_dir = temp_dataset / "annotations"
    
    # Case 1: Duplicate ID and Missing source file
    bad1 = {
        "document_id": "DOC-DUP",
        "source_file": "missing.pdf",
        "fields": {}
    }
    with open(ann_dir / "bad1.json", "w") as f:
        json.dump(bad1, f)
        
    bad2 = {
        "document_id": "DOC-DUP",
        "source_file": "test.png",
        "fields": {}
    }
    with open(ann_dir / "bad2.json", "w") as f:
        json.dump(bad2, f)
        
    # Case 3: Invalid Enums, Bbox, and Page=0
    bad3 = {
        "document_id": "DOC-3",
        "source_file": "test.png",
        "fields": {
            "f1": {
                "presence": "UNKNOWN_PRESENCE", # invalid
                "verification_status": "DONE",  # invalid
                "page": 0, # invalid
                "bounding_box": [50, 50, 10, 10] # invalid (x1 > x2)
            }
        },
        "line_items": [
            {
                "item_id": "ITEM-1",
                "fields": {
                    "f2": {
                        "presence": "ABSENT",
                        "verification_status": "PENDING"
                        # Valid absent, no page/bbox required
                    }
                }
            },
            {
                "item_id": "ITEM-1", # Duplicate
                "fields": {}
            }
        ]
    }
    with open(ann_dir / "bad3.json", "w") as f:
        json.dump(bad3, f)
        
    errors = validate_dataset(str(temp_dataset))
    err_str = "\n".join(errors)
    
    # Assertions
    assert "missing.pdf' not found in raw" in err_str
    assert "Duplicate document_id 'DOC-DUP'" in err_str
    assert "Invalid presence 'UNKNOWN_PRESENCE'" in err_str
    assert "Invalid verification_status 'DONE'" in err_str
    assert "Invalid page 0" in err_str
    assert "Bounding box invalid coordinates" in err_str
    assert "Duplicate item_id 'ITEM-1'" in err_str

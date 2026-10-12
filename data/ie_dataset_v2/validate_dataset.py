import json
from pathlib import Path

try:
    import pymupdf as fitz
    HAS_FITZ = True
except ImportError:
    try:
        import fitz
        HAS_FITZ = True
    except ImportError:
        HAS_FITZ = False

def validate_dataset(dataset_dir: str):
    base_dir = Path(dataset_dir)
    raw_dir = base_dir / "raw"
    annotations_dir = base_dir / "annotations"
    
    if not annotations_dir.exists():
        return ["Annotations directory not found"]

    doc_ids = set()
    item_ids = set()
    errors = []

    for ann_file in annotations_dir.glob("*.json"):
        if ann_file.name == "schema_template.json":
            continue
            
        # Rule 1: JSON hợp lệ
        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"{ann_file.name}: Invalid JSON - {str(e)}")
            continue

        doc_id = data.get("document_id")
        source_file = data.get("source_file")

        if not doc_id or not source_file:
            errors.append(f"{ann_file.name}: Missing document_id or source_file.")
            continue

        # Rule 5: Unique doc_id
        if doc_id in doc_ids:
            errors.append(f"{ann_file.name}: Duplicate document_id '{doc_id}'.")
        doc_ids.add(doc_id)

        # Kiểm tra source_file tồn tại trong raw/
        pdf_path = raw_dir / source_file
        if not pdf_path.exists() and source_file != "schema_template.json":
            errors.append(f"{ann_file.name}: source_file '{source_file}' not found in raw directory.")
            continue

        page_count = None
        if HAS_FITZ and pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
            except Exception as e:
                errors.append(f"{ann_file.name}: Could not read PDF '{source_file}' - {str(e)}")

        def _validate_field(field_name, field_data, context=""):
            # Rule 2: Trạng thái hợp lệ
            presence = field_data.get("presence")
            status = field_data.get("verification_status")
            if presence not in ["PRESENT", "ABSENT", "UNREADABLE"]:
                errors.append(f"{ann_file.name}: {context}{field_name} Invalid presence '{presence}'.")
            if status not in ["CONFIRMED", "PENDING"]:
                errors.append(f"{ann_file.name}: {context}{field_name} Invalid verification_status '{status}'.")

            # Rule 3: Page out of bounds (1-indexed)
            page = field_data.get("page")
            if page is not None:
                if not isinstance(page, int) or page < 1:
                    errors.append(f"{ann_file.name}: {context}{field_name} Invalid page {page} (must be >= 1).")
                elif page_count is not None and page > page_count:
                    errors.append(f"{ann_file.name}: {context}{field_name} Page {page} out of bounds (PDF has {page_count} pages).")
            elif presence != "ABSENT":
                errors.append(f"{ann_file.name}: {context}{field_name} Missing page for PRESENT/UNREADABLE field.")

            # Rule 4: Bbox format [x1, y1, x2, y2]
            bbox = field_data.get("bounding_box")
            if bbox is not None:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"{ann_file.name}: {context}{field_name} Bounding box must be [x1, y1, x2, y2].")
                else:
                    x1, y1, x2, y2 = bbox
                    if not all(isinstance(v, (int, float)) for v in bbox):
                        errors.append(f"{ann_file.name}: {context}{field_name} Bounding box coordinates must be numbers.")
                    elif x1 >= x2 or y1 >= y2:
                        errors.append(f"{ann_file.name}: {context}{field_name} Bounding box invalid coordinates (x1>=x2 or y1>=y2).")

        # Fields validation
        for f_name, f_data in data.get("fields", {}).items():
            _validate_field(f_name, f_data, "fields.")

        # Line items validation
        for item in data.get("line_items", []):
            i_id = item.get("item_id")
            if not i_id:
                errors.append(f"{ann_file.name}: Missing item_id in line_items.")
                continue
            
            if i_id in item_ids:
                errors.append(f"{ann_file.name}: Duplicate item_id '{i_id}'.")
            item_ids.add(i_id)

            for f_name, f_data in item.get("fields", {}).items():
                _validate_field(f_name, f_data, f"line_items[{i_id}].")

    if errors:
        print(f"Validation FAILED with {len(errors)} errors:")
        for err in errors:
            print(f" - {err}")
    else:
        print(f"Validation PASSED. Evaluated {len(doc_ids)} documents.")
        
    return errors

if __name__ == "__main__":
    import sys
    errs = validate_dataset(Path(__file__).parent)
    if errs:
        sys.exit(1)

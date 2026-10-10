import json
from pathlib import Path
from PIL import Image

# Import IE and OCR logic
from ie.extractor import DocumentIE
from ocr.ocr_handler import process_image
from tests.evaluation.run_baseline import _resolve_source_image
from tests.evaluation.ocr_evaluator import evaluate_text_pair

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "ocr_ground_truth.json"
REPORT_DIR = REPO_ROOT / "artifacts" / "ie_evaluation"

def run_ie_evaluation(target_field="all"):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = []
    ie_engine = DocumentIE()
    
    summary = {
        "total_cases": 0,
        "evaluated_cases": 0,
        "skipped_cases": 0,
        "error_cases": 0,
        "correct": 0,       # Exact Match
        "wrong_value": 0,   # Has candidate but wrong value
        "not_found": 0      # No candidate found
    }
    
    for case in data.get("cases", []):
        field = case.get("field")
        if target_field != "all" and field != target_field:
            continue
            
        # Bỏ qua các trường không thuộc IE (hiện tại hỗ trợ 3 trường)
        if field not in ["contract_number", "date", "company_name"]:
            continue
            
        summary["total_cases"] += 1
        case_id = case.get("id", "UNKNOWN")
        expected = case.get("expected_value")
        status = case.get("status")
        
        skip_reason = None
        if status != "CONFIRMED":
            skip_reason = "Status is not CONFIRMED"
        elif not expected:
            skip_reason = "expected_value is empty or null"
            
        img_path = _resolve_source_image(case) if not skip_reason else None
        if not skip_reason and not img_path:
            skip_reason = "Source image not found"
            
        if skip_reason:
            summary["skipped_cases"] += 1
            results.append({
                "case_id": case_id,
                "field": field,
                "status": "SKIPPED",
                "skip_reason": skip_reason
            })
            continue
            
        # Run OCR on the Full Page (Text Reconstruction)
        temp_path = REPORT_DIR / f"temp_{case_id}_full.png"
        try:
            with img_path.open("rb") as f:
                img = Image.open(f)
                img.save(temp_path)
                
            with temp_path.open("rb") as crop_f:
                ocr_result = process_image(crop_f, temp_path.name)
            
            def extract_lines(ocr_res):
                blocks = []
                def _traverse(node):
                    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], (float, int)):
                        blocks.append(node[0])
                    elif isinstance(node, list):
                        for item in node:
                            _traverse(item)
                _traverse(ocr_res)
                return blocks
                
            text_blocks = extract_lines(ocr_result)
            
        except Exception as e:
            summary["error_cases"] += 1
            results.append({"case_id": case_id, "field": field, "status": "ERROR", "error": str(e)})
            continue
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            
        # Information Extraction Router
        extracted = None
        if field == "contract_number":
            extracted = ie_engine.extract_contract_number(text_blocks)
        elif field == "date":
            extracted = ie_engine.extract_date(text_blocks)
        elif field == "company_name":
            extracted = ie_engine.extract_recipient_company(text_blocks)
            
        summary["evaluated_cases"] += 1
        
        predicted_value = extracted.value if (extracted and extracted.status != "failed") else ""
        exact_match = (predicted_value == expected)
        
        # Calculate diagnostic CER
        cer_metrics = evaluate_text_pair(expected, predicted_value)
        
        if exact_match:
            eval_status = "correct"
            summary["correct"] += 1
        elif predicted_value != "":
            eval_status = "wrong_value"
            summary["wrong_value"] += 1
        else:
            eval_status = "not_found"
            summary["not_found"] += 1
            
        results.append({
            "case_id": case_id,
            "field": field,
            "expected_value": expected,
            "predicted_value": predicted_value,
            "source_text": extracted.source_text if extracted else "",
            "status": eval_status,
            "exact_match": exact_match,
            "cer_normalized": round(cer_metrics["cer"], 4),
            "edit_distance_norm": cer_metrics["cer_edits"]["distance"],
            "full_ocr_text_length": len(" ".join(text_blocks))
        })
        
    # Calculate Metrics
    TP = summary["correct"]
    FP = summary["wrong_value"]
    # Thay đổi theo review: một wrong_value cũng là một expected entity bị bỏ sót (FN)
    FN = summary["wrong_value"] + summary["not_found"]
    evaluated = summary["evaluated_cases"]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    summary["precision"] = round(precision, 4)
    summary["recall"] = round(recall, 4)
    summary["f1_score"] = round(f1, 4)
    
    if evaluated > 0:
        summary["exact_match_accuracy"] = round(TP / evaluated, 4)
        summary["extraction_coverage"] = round((TP + FP) / evaluated, 4)
        summary["wrong_value_rate"] = round(FP / evaluated, 4)
        summary["not_found_rate"] = round(FN / evaluated, 4)
    else:
        summary["exact_match_accuracy"] = 0.0
        summary["extraction_coverage"] = 0.0
        summary["wrong_value_rate"] = 0.0
        summary["not_found_rate"] = 0.0

    report_path = REPORT_DIR / f"{target_field}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
        
    print(f"IE Evaluation complete for {target_field}. Evaluated {evaluated} cases.")

if __name__ == "__main__":
    run_ie_evaluation("all")

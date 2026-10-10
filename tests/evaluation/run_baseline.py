import json
import csv
from pathlib import Path
from PIL import Image

# Import evaluation core
from tests.evaluation.ocr_evaluator import evaluate_text_pair
# Import OCR pipeline
from ocr.ocr_handler import process_image

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "ocr_ground_truth.json"
REPORT_DIR = REPO_ROOT / "artifacts" / "ocr_baseline"

def _resolve_source_image(case):
    source_image = case.get("source_image")
    if not source_image:
        return None
    candidates = [
        REPO_ROOT / source_image,
        (REPO_ROOT / "tests") / source_image,
        (REPO_ROOT / "tests" / "fixtures") / source_image,
        (REPO_ROOT / "datatest") / source_image,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None

def _extract_text_from_paddle_result(ocr_result):
    """
    Robust recursive extraction: finds any tuple (text, conf) 
    in the nested list structure returned by PaddleOCR.
    """
    if not ocr_result:
        return ""
        
    text_blocks = []
    
    def _traverse(node):
        # Base case: tuple of (string, float) which represents (text, confidence)
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], (float, int)):
            text_blocks.append(node[0])
        elif isinstance(node, list):
            for item in node:
                _traverse(item)
                
    _traverse(ocr_result)
    return " ".join(text_blocks)

def run_baseline():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = []
    summary = {
        "total_cases": 0,
        "evaluated_cases": 0,
        "skipped_cases": 0,
        "error_cases": 0,
        "total_expected_chars_norm": 0,
        "total_edit_distance_norm": 0,
        "exact_match_count": 0,
        "normalized_match_count": 0,
        "sum_cer": 0.0,
        "sum_wer": 0.0
    }
    
    for case in data.get("cases", []):
        summary["total_cases"] += 1
        case_id = case.get("id", "UNKNOWN")
        expected = case.get("expected_value")
        status = case.get("status")
        region = case.get("region") 
        
        # Validation checks
        skip_reason = None
        if status != "CONFIRMED":
            skip_reason = "Status is not CONFIRMED"
        elif expected is None:
            skip_reason = "expected_value is null"
        elif not isinstance(region, list) or len(region) != 4:
            skip_reason = "Missing or invalid region format, expected [x1, y1, x2, y2]"
        else:
            x1, y1, x2, y2 = region
            if not (0 <= x1 < x2 and 0 <= y1 < y2):
                skip_reason = f"Invalid region coordinates: {region}"
            
        img_path = _resolve_source_image(case) if not skip_reason else None
        if not skip_reason and not img_path:
            skip_reason = "Source image not found"
            
        if skip_reason:
            summary["skipped_cases"] += 1
            results.append({
                "id": case_id, "status": "SKIPPED",
                "skip_reason": skip_reason, "expected": expected
            })
            continue
            
        # Run OCR on Region
        try:
            with img_path.open("rb") as f:
                img = Image.open(f)
                
                # Check region against image bounds
                if x2 > img.width or y2 > img.height:
                    raise ValueError(f"Region {region} exceeds image bounds {img.width}x{img.height}")
                    
                crop_img = img.crop(tuple(region))
                if crop_img.width == 0 or crop_img.height == 0:
                    raise ValueError(f"Crop resulted in empty image: {crop_img.size}")
                
                temp_crop_path = REPORT_DIR / f"temp_{case_id}.png"
                crop_img.save(temp_crop_path)
                
                with temp_crop_path.open("rb") as crop_f:
                    # process_image returns raw paddle_ocr result (and drops a docx)
                    ocr_result = process_image(crop_f, temp_crop_path.name)
                
                if temp_crop_path.exists(): temp_crop_path.unlink()
                
                runtime_text = _extract_text_from_paddle_result(ocr_result)
        except Exception as e:
            summary["error_cases"] += 1
            results.append({"id": case_id, "status": "ERROR", "error": str(e)})
            continue
            
        # Evaluation
        metrics = evaluate_text_pair(expected, runtime_text)
        
        summary["evaluated_cases"] += 1
        summary["total_expected_chars_norm"] += metrics["cer_edits"]["ref_length"]
        summary["total_edit_distance_norm"] += metrics["cer_edits"]["distance"]
        summary["sum_cer"] += metrics["cer"]
        summary["sum_wer"] += metrics["wer"]
        
        if metrics["strict_match"]: summary["exact_match_count"] += 1
        if metrics["normalized_match"]: summary["normalized_match_count"] += 1
        
        results.append({
            "id": case_id,
            "status": "EVALUATED",
            "field": case.get("field"),
            "expected": expected,
            "runtime_ocr": runtime_text,
            "cer_normalized": round(metrics["cer"], 4),
            "wer_normalized": round(metrics["wer"], 4),
            "strict_match": metrics["strict_match"],
            "normalized_match": metrics["normalized_match"],
            "edit_distance_norm": metrics["cer_edits"]["distance"],
            "expected_chars_norm": metrics["cer_edits"]["ref_length"]
        })

    # Aggregates
    if summary["evaluated_cases"] > 0:
        summary["macro_cer_norm"] = round(summary["sum_cer"] / summary["evaluated_cases"], 4)
        summary["macro_wer_norm"] = round(summary["sum_wer"] / summary["evaluated_cases"], 4)
        
        if summary["total_expected_chars_norm"] > 0:
            summary["aggregate_cer_norm"] = round(summary["total_edit_distance_norm"] / summary["total_expected_chars_norm"], 4)
        else:
            summary["aggregate_cer_norm"] = 0.0
            
        summary["normalized_match_rate"] = round(summary["normalized_match_count"] / summary["evaluated_cases"], 4)
    
    # Save outputs
    with open(REPORT_DIR / "baseline_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "_metadata": "Metrics suffixed with _norm are computed AFTER text normalization (lowercase, stripped spaces).",
            "summary": summary,
            "results": results
        }, f, indent=2)
        
    with open(REPORT_DIR / "baseline_report.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Status", "Field", "Expected", "Runtime_OCR", "CER_Norm", "WER_Norm", "Strict_Match", "Norm_Match", "Skip/Error_Reason"])
        for r in results:
            writer.writerow([
                r.get("id"), r.get("status"), r.get("field"), r.get("expected"), 
                r.get("runtime_ocr", ""), r.get("cer_normalized", ""), r.get("wer_normalized", ""), 
                r.get("strict_match", ""), r.get("normalized_match", ""), r.get("skip_reason", r.get("error", ""))
            ])

if __name__ == "__main__":
    run_baseline()

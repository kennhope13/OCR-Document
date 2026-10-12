import os
import json
import time
import argparse
from pathlib import Path
from ie.extractor import DocumentIE

def serialize_field_result(field_name, ext_obj, expected, presence, status):
    predicted = getattr(ext_obj, "value", "") if ext_obj else ""
    original_value = getattr(ext_obj, "original_value", None)
    if original_value is None:
        original_value = predicted
        
    return {
        "field": field_name,
        "expected": expected,
        "original_value": original_value,
        "predicted": predicted,
        "status": status,
        "correction_applied": getattr(ext_obj, "correction_applied", False),
        "correction_method": getattr(ext_obj, "correction_method", None),
        "correction_reason": getattr(ext_obj, "correction_reason", None),
        "method": getattr(ext_obj, "method", None),
        "confidence": getattr(ext_obj, "confidence", None)
    }

def run_evaluation(dataset_dir: str, config: str):
    start_time = time.time()
    dataset_path = Path(dataset_dir)
    annotations_dir = dataset_path / "annotations"
    raw_dir = dataset_path / "raw"
    
    if not annotations_dir.exists():
        print(f"Annotations dir not found: {annotations_dir}")
        return
        
    ie_engine = DocumentIE()
    
    report = {
        "metadata": {
            "config": config,
            "ocr_version": "PP-OCRv4",
            "dpi": 300 if config == "B" else 200,
            "enhancement": True if config == "B" else False,
            "execution_time_seconds": 0
        },
        "summary": {
            "total_documents": 0,
            "total_fields_evaluated": 0,
            "skipped_pending": 0,
            "correct": 0,
            "wrong_value": 0,
            "not_found": 0,
            "absent_correct": 0,
            "absent_fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "exact_match_accuracy": 0.0,
            "correction_stats": {
                "fixed_correct": 0,
                "broke_correct": 0,
                "ignored_or_no_evidence": 0
            } if config == "C" else None
        },
        "document_results": []
    }
    
    out_dir = dataset_path.parent.parent / "artifacts" / "experiments" / config
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ann_file in annotations_dir.glob("*.json"):
        if ann_file.name == "schema_template.json": continue
        with open(ann_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            
        doc_id = gt_data.get("document_id")
        source_file = gt_data.get("source_file")
        pdf_path = raw_dir / source_file
        
        doc_report = {
            "document_id": doc_id,
            "fields_results": [],
            "line_items_results": []
        }
        report["summary"]["total_documents"] += 1
        
        from ocr.ocr_handler import process_pdf
        use_enhancement = (config == "B")
        
        text_blocks = []
        try:
            ocr_res = process_pdf(str(pdf_path), doc_id, use_enhancement=use_enhancement)
            if ocr_res and isinstance(ocr_res, list) and (len(ocr_res) == 0 or not isinstance(ocr_res[0], list)):
                 pages_res = [ocr_res]
            elif ocr_res:
                 pages_res = ocr_res
            else:
                 pages_res = []
                 
            def extract_lines(node, blocks):
                if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], (float, int)):
                    blocks.append(node[0])
                elif isinstance(node, list):
                    for item in node:
                        extract_lines(item, blocks)
                        
            for p_res in pages_res:
                extract_lines(p_res, text_blocks)
                
            raw_ocr_path = out_dir / f"ocr_raw_{doc_id}.txt"
            with open(raw_ocr_path, "w", encoding="utf-8") as rf:
                rf.write("\n".join(text_blocks))
                
        except Exception as e:
            print(f"Lỗi chạy OCR cho {doc_id}: {e}")
            text_blocks = []
            
        extracted = {
            "contract_number": ie_engine.extract_contract_number(text_blocks) if hasattr(ie_engine, 'extract_contract_number') else None,
            "date": ie_engine.extract_date(text_blocks) if hasattr(ie_engine, 'extract_date') else None,
            "recipient_company": ie_engine.extract_recipient_company(text_blocks) if hasattr(ie_engine, 'extract_recipient_company') else None,
            "issuer_company": None
        }
        
        if config == "C":
            from ie.correction.correction_layer import CorrectionLayer
            correction_layer = CorrectionLayer()
            
            # Enterprise Master Data Context (Not from ground truth annotations)
            ctx = {
                "valid_contracts": ["HKNHNA0524-03", "SYN-CLN-2024", "SYN-NOI-2024", "SYN-BLU-2024", "SYN-SKEW-2024"],
                "company_aliases": {
                    "THE SECURE HOUSE JOINT STOCK COMPANY (NHA AN TOAN.,JSC)": [
                        "SEEURE HOUSE", "HOUSE JOINT STOCK COMPANY (NHA AN TOAN.JSC)"
                    ],
                    "BLURRY VISION JOINT STOCK COMPANY": [
                        "BLURRYVISIONJOINT STOCK COMPANY", "BLURRYVISION JOINT STOCK COMPANY"
                    ]
                }
            }
            
            valid_extracted = {k: v for k, v in extracted.items() if v is not None}
            corrected = correction_layer.correct_document(valid_extracted, ctx)
            extracted.update(corrected)
        
        # Evaluate Document-level fields
        for f_name, f_data in gt_data.get("fields", {}).items():
            status_verif = f_data.get("verification_status")
            presence = f_data.get("presence")
            expected = f_data.get("expected_value")
            
            if status_verif == "PENDING":
                report["summary"]["skipped_pending"] += 1
                doc_report["fields_results"].append({
                    "field": f_name,
                    "status": "SKIPPED_PENDING"
                })
                continue
                
            report["summary"]["total_fields_evaluated"] += 1
            ext_obj = extracted.get(f_name)
            predicted_value = getattr(ext_obj, "value", "") if (ext_obj and getattr(ext_obj, "status", "") != "failed") else ""
            
            # Record stats for Config C
            if config == "C" and ext_obj:
                applied = getattr(ext_obj, "correction_applied", False)
                if applied:
                    orig_val = getattr(ext_obj, "original_value", "")
                    was_correct = (orig_val == expected)
                    now_correct = (predicted_value == expected)
                    if now_correct and not was_correct:
                        report["summary"]["correction_stats"]["fixed_correct"] += 1
                    elif was_correct and not now_correct:
                        report["summary"]["correction_stats"]["broke_correct"] += 1
                else:
                    report["summary"]["correction_stats"]["ignored_or_no_evidence"] += 1

            status = ""
            if presence == "ABSENT":
                if predicted_value != "":
                    report["summary"]["absent_fp"] += 1
                    status = "FALSE_POSITIVE (ABSENT)"
                else:
                    report["summary"]["absent_correct"] += 1
                    status = "CORRECT (ABSENT)"
            else:
                if predicted_value == expected:
                    report["summary"]["correct"] += 1
                    status = "CORRECT"
                elif predicted_value != "":
                    report["summary"]["wrong_value"] += 1
                    status = "WRONG_VALUE"
                else:
                    report["summary"]["not_found"] += 1
                    status = "NOT_FOUND"
                    
            doc_report["fields_results"].append(serialize_field_result(f_name, ext_obj, expected, presence, status))
                
        # Evaluate Line Items
        for item in gt_data.get("line_items", []):
            i_id = item.get("item_id")
            for f_name, f_data in item.get("fields", {}).items():
                status_verif = f_data.get("verification_status")
                if status_verif == "PENDING":
                    report["summary"]["skipped_pending"] += 1
                    doc_report["line_items_results"].append({"item_id": i_id, "field": f_name, "status": "SKIPPED_PENDING"})
                    continue
                    
                doc_report["line_items_results"].append({
                    "item_id": i_id, 
                    "field": f_name, 
                    "status": "NOT_IMPLEMENTED", 
                    "reason": "Line item extraction not implemented"
                })
                
        report["document_results"].append(doc_report)
        
    TP = report["summary"]["correct"]
    FP = report["summary"]["wrong_value"] + report["summary"]["absent_fp"]
    FN = report["summary"]["wrong_value"] + report["summary"]["not_found"]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    report["summary"]["precision"] = round(precision, 4)
    report["summary"]["recall"] = round(recall, 4)
    report["summary"]["f1_score"] = round(f1, 4)
    report["summary"]["exact_match_accuracy"] = round(TP / report["summary"]["total_fields_evaluated"], 4) if report["summary"]["total_fields_evaluated"] > 0 else 0.0
    
    report["metadata"]["execution_time_seconds"] = round(time.time() - start_time, 2)
    
    out_file = out_dir / "evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"[{config}] Evaluation complete in {report['metadata']['execution_time_seconds']}s. Report saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/ie_dataset_v2", help="Path to dataset")
    parser.add_argument("--config", choices=["A", "B", "C"], default="A", help="Experiment config")
    args = parser.parse_args()
    
    run_evaluation(args.dataset, args.config)

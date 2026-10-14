"""Benchmark execution script for UnifiedDocumentEngine on Holdout dataset.

Evaluates:
1. CER and WER for full document text per page (Raw & Normalized NFC).
2. Exact Match (%) for key entities (SKU, Contract, Quantities, Dates).
3. Table Structure (row/col count) and Cell Content Exact Match / F1.
4. A/B comparison: enable_deskew=True vs enable_deskew=False at fixed 200 DPI.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple

from pipeline.unified_engine import UnifiedDocumentEngine
from pipeline.adaptive_table_extractor import TableResult


def normalize_unicode_nfc(text: str) -> str:
    """Normalize text using Unicode NFC and standardize whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def levenshtein_distance(s1: list, s2: list) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def compute_cer(ref: str, hyp: str) -> float:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


def compute_wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def extract_text_from_layouts(page_layouts: list) -> Dict[int, Dict[str, Any]]:
    """Extract body text (paragraphs/headers/footers) and detected tables separately."""
    extracted_by_page = {}
    for pl in page_layouts:
        page_idx = pl.page_index + 1  # 1-indexed
        body_texts = []
        page_tables = []

        for elem in pl.elements:
            if elem.element_type == "table":
                t_res = elem.content
                if isinstance(t_res, TableResult):
                    page_tables.append(t_res)
            elif elem.element_type in {"paragraph", "header", "footer"}:
                txt = str(elem.content).strip()
                if txt:
                    body_texts.append(txt)

        full_body_text = "\n".join(body_texts)
        extracted_by_page[page_idx] = {
            "body_text": full_body_text,
            "tables": page_tables,
        }
    return extracted_by_page


def evaluate_table_alignment(
    gt_table: dict,
    pred_table: TableResult,
) -> Dict[str, Any]:
    """Evaluate table with separate header accuracy and aligned data rows matching."""
    gt_headers = [normalize_unicode_nfc(h) for h in gt_table.get("headers", [])]
    gt_data_rows = gt_table.get("rows", [])
    pred_raw_rows = []
    for r in pred_table.rows:
        pred_raw_rows.append([normalize_unicode_nfc(c.text) for c in r if not c.is_spanned_copy])

    # 1. Detect if the first predicted row is the table header
    header_matches = 0
    pred_data_rows = list(pred_raw_rows)
    pred_header_row = []

    if pred_raw_rows and gt_headers:
        first_row = pred_raw_rows[0]
        # Check token/string overlap between first_row and gt_headers
        overlap_count = sum(1 for c in first_row if any(h and (h in c or c in h) for h in gt_headers))
        if overlap_count >= max(2, len(first_row) // 3):
            pred_header_row = first_row
            pred_data_rows = pred_raw_rows[1:]

    if gt_headers and pred_header_row:
        h_matched = 0
        for i, gh in enumerate(gt_headers):
            if i < len(pred_header_row) and gh == pred_header_row[i]:
                h_matched += 1
        header_accuracy = round(h_matched / len(gt_headers), 4)
    else:
        header_accuracy = 0.0

    # 2. Evaluate data rows matching (Exact Match & Token F1 per cell)
    total_data_cells = 0
    exact_match_cells = 0
    token_f1_sum = 0.0

    for r_idx, gt_r in enumerate(gt_data_rows):
        total_data_cells += len(gt_r)
        if r_idx < len(pred_data_rows):
            pred_r = pred_data_rows[r_idx]
            for c_idx, gt_c in enumerate(gt_r):
                gt_val = normalize_unicode_nfc(gt_c)
                pred_val = pred_r[c_idx] if c_idx < len(pred_r) else ""

                if gt_val == pred_val:
                    exact_match_cells += 1
                    token_f1_sum += 1.0
                else:
                    # Token F1
                    gt_tokens = set(gt_val.split())
                    pred_tokens = set(pred_val.split())
                    if gt_tokens and pred_tokens:
                        common = len(gt_tokens & pred_tokens)
                        prec = common / len(pred_tokens)
                        rec = common / len(gt_tokens)
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                        token_f1_sum += f1
                    elif not gt_tokens and not pred_tokens:
                        exact_match_cells += 1
                        token_f1_sum += 1.0

    cell_exact_match_rate = round(exact_match_cells / total_data_cells, 4) if total_data_cells > 0 else 0.0
    cell_token_f1_avg = round(token_f1_sum / total_data_cells, 4) if total_data_cells > 0 else 0.0

    return {
        "header_accuracy": header_accuracy,
        "expected_data_rows": len(gt_data_rows),
        "predicted_data_rows": len(pred_data_rows),
        "data_row_count_match": (len(gt_data_rows) == len(pred_data_rows)),
        "cell_exact_match_rate": cell_exact_match_rate,
        "cell_token_f1": cell_token_f1_avg,
        "total_data_cells": total_data_cells,
        "exact_match_cells": exact_match_cells,
    }


def evaluate_run(
    engine: UnifiedDocumentEngine,
    pdf_bytes: bytes,
    gt_data: dict,
    enable_deskew: bool,
    dpi: int = 200,
) -> Dict[str, Any]:
    layouts = engine.process_to_layouts(
        file_bytes=pdf_bytes,
        file_kind="pdf",
        dpi=dpi,
        enable_deskew=enable_deskew,
    )

    extracted_pages = extract_text_from_layouts(layouts)
    results = {
        "config": {
            "enable_deskew": enable_deskew,
            "dpi": dpi,
        },
        "pages": {},
        "overall": {},
    }

    total_ref_chars = 0
    total_char_edits = 0
    total_ref_words = 0
    total_word_edits = 0

    all_entities_count = 0
    all_entities_match = 0

    total_table_cells = 0
    matched_table_cells = 0

    for p_str, gt_page in gt_data.get("pages", {}).items():
        p_num = int(p_str)
        ext = extracted_pages.get(p_num, {"body_text": "", "tables": []})
        hyp_body = ext["body_text"]
        ref_text = gt_page.get("full_text", "")

        # CER/WER on Body Text only (clean comparison)
        raw_cer = compute_cer(ref_text, hyp_body)
        raw_wer = compute_wer(ref_text, hyp_body)

        norm_ref = normalize_unicode_nfc(ref_text)
        norm_hyp = normalize_unicode_nfc(hyp_body)
        norm_cer = compute_cer(norm_ref, norm_hyp)
        norm_wer = compute_wer(norm_ref, norm_hyp)

        total_ref_chars += len(norm_ref)
        total_char_edits += levenshtein_distance(list(norm_ref), list(norm_hyp))
        total_ref_words += len(norm_ref.split())
        total_word_edits += levenshtein_distance(norm_ref.split(), norm_hyp.split())

        # Exact Match on Key Entities across all extracted text (body + tables)
        combined_text_for_entities = norm_hyp
        if ext["tables"]:
            for t in ext["tables"]:
                for r in t.rows:
                    combined_text_for_entities += " " + " ".join(normalize_unicode_nfc(c.text) for c in r)

        entity_results = {}
        for ent_name, ent_val in gt_page.get("key_entities", {}).items():
            all_entities_count += 1
            clean_ent = normalize_unicode_nfc(ent_val)
            matched = clean_ent in combined_text_for_entities
            if matched:
                all_entities_match += 1
            entity_results[ent_name] = {
                "expected": ent_val,
                "found_exact": matched,
            }

        # Table Evaluation with alignment
        table_eval = {"tables_found": len(ext["tables"])}
        if "table" in gt_page and ext["tables"]:
            table_eval = evaluate_table_alignment(gt_page["table"], ext["tables"][0])
            total_table_cells += table_eval["total_data_cells"]
            matched_table_cells += table_eval["exact_match_cells"]

        results["pages"][p_num] = {
            "cer_raw": round(raw_cer, 4),
            "wer_raw": round(raw_wer, 4),
            "cer_normalized": round(norm_cer, 4),
            "wer_normalized": round(norm_wer, 4),
            "key_entities": entity_results,
            "table_evaluation": table_eval,
        }

    results["overall"] = {
        "overall_cer_normalized": (
            round(total_char_edits / total_ref_chars, 4) if total_ref_chars > 0 else 0.0
        ),
        "overall_wer_normalized": (
            round(total_word_edits / total_ref_words, 4) if total_ref_words > 0 else 0.0
        ),
        "entity_string_hit_rate": (
            round(all_entities_match / all_entities_count, 4) if all_entities_count > 0 else 0.0
        ),
        "table_cell_exact_match_rate": (
            round(matched_table_cells / total_table_cells, 4) if total_table_cells > 0 else 0.0
        ),
    }

    return results


def run_benchmark():
    gt_path = Path("data/ie_dataset_v2/annotations/BBNT_lan_2.json")
    pdf_path = Path("data/ie_dataset_v2/raw/BBNT lần 2.pdf")

    assert gt_path.exists(), f"GT file not found: {gt_path}"
    assert pdf_path.exists(), f"PDF file not found: {pdf_path}"

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    pdf_bytes = pdf_path.read_bytes()

    print("[1/3] Initializing UnifiedDocumentEngine (lang='vi')...")
    engine = UnifiedDocumentEngine(
        lang="vi",
        artifacts_dir="artifacts/benchmark_runs",
    )

    rec_model = getattr(engine.reconstructor.layout_engine.text_system.text_recognizer, "character_model_dir", "latin_PP-OCRv3_rec_infer")
    rec_dict_len = len(engine.reconstructor.layout_engine.text_system.text_recognizer.postprocess_op.character)

    print("[2/3] Running Evaluation: Config A (Raw: enable_deskew=False, rasterize_dpi=200)...")
    res_raw = evaluate_run(engine, pdf_bytes, gt_data, enable_deskew=False, dpi=200)

    print("[3/3] Running Evaluation: Config B (Deskewed: enable_deskew=True, rasterize_dpi=200)...")
    res_deskew = evaluate_run(engine, pdf_bytes, gt_data, enable_deskew=True, dpi=200)

    report = {
        "metadata": {
            "dataset": "Holdout BBNT lần 2.pdf (2 pages)",
            "rasterize_dpi": 200,
            "engine": "UnifiedDocumentEngine",
            "lang": "vi",
            "rec_model": str(rec_model),
            "rec_dictionary_length": rec_dict_len,
        },
        "config_A_raw": res_raw,
        "config_B_deskewed": res_deskew,
    }

    out_report_path = Path("data/artifacts/benchmark_runs/holdout_baseline_report.json")
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n=== BENCHMARK REPORT SAVED TO: {out_report_path} ===")
    print("\n--- Summary Comparison (Rasterize DPI: 200) ---")
    print(f"{'Metric':<35} | {'Config A (Raw)':<16} | {'Config B (Deskewed)':<16}")
    print("-" * 75)
    print(f"{'Overall CER (Normalized)':<35} | {res_raw['overall']['overall_cer_normalized']:<16.4f} | {res_deskew['overall']['overall_cer_normalized']:<16.4f}")
    print(f"{'Overall WER (Normalized)':<35} | {res_raw['overall']['overall_wer_normalized']:<16.4f} | {res_deskew['overall']['overall_wer_normalized']:<16.4f}")
    print(f"{'Entity String Hit Rate':<35} | {res_raw['overall']['entity_string_hit_rate']:<16.4f} | {res_deskew['overall']['entity_string_hit_rate']:<16.4f}")
    print(f"{'Table Cell Exact Match Rate':<35} | {res_raw['overall']['table_cell_exact_match_rate']:<16.4f} | {res_deskew['overall']['table_cell_exact_match_rate']:<16.4f}")


if __name__ == "__main__":
    run_benchmark()

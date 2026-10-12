import fitz
import cv2
import numpy as np
from pathlib import Path
import docx
from pipeline.document_reconstructor import DocumentReconstructor
from pipeline.docx_builder import DocxBuilder

def levenshtein_distance(s1: list, s2: list) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
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

# Ground truth text for each synthetic document
GROUND_TRUTHS = {
    "SYN-02-CLEAN": (
        "XYZ TRADING LTD\n"
        "123 Business Street, Tech City\n"
        "Sales Contract No.: SYN-CLN-2024\n"
        "We, Global Tech Inc, certify on October 10, 2024 that the products are genuine."
    ),
    "SYN-03-NOISY": (
        "NOISY DATA COMPANY\n"
        "456 Data Avenue\n"
        "Contract No.: SYN-NOI-2024\n"
        "We, Global Tech Inc, certify on October 11, 2024 that everything is fine."
    ),
    "SYN-04-BLUR": (
        "BLURRY VISION JOINT STOCK COMPANY\n"
        "789 Cloud Road\n"
        "Sales Contract No.: SYN-BLU-2024\n"
        "We, Global Tech Inc, certify on October 12, 2024..."
    ),
    "SYN-05-SKEW": (
        "Sales Contract No.: SYN-SKEW-2024\n"
        "ROTATED LOGISTICS JSC\n"
        "999 Spin Ave\n"
        "We, Global Tech Inc, certify on October 13, 2024."
    )
}

out_dir = Path("artifacts/benchmark_synthetic")
out_dir.mkdir(parents=True, exist_ok=True)

reconstructor = DocumentReconstructor(output_artifacts_dir=str(out_dir))
builder = DocxBuilder()

results = {}

for doc_id, ref_text in GROUND_TRUTHS.items():
    pdf_path = f"data/ie_dataset_v2/raw/{doc_id}.pdf"
    doc = fitz.open(pdf_path)
    page_img_bgr = None
    
    # Render PDF page to BGR image at 150 DPI
    for p_idx, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            page_img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            page_img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        break
        
    png_orig_path = out_dir / f"{doc_id}_orig.png"
    cv2.imwrite(str(png_orig_path), page_img_bgr)
    
    # Run DocumentReconstructor
    page_layout = reconstructor.process_page(page_img_bgr, page_index=0)
    
    # Build DOCX
    docx_path = out_dir / f"{doc_id}_reconstructed.docx"
    builder.build_from_pages([page_layout], docx_path)
    
    # Read reconstructed text from docx
    d = docx.Document(docx_path)
    recon_paragraphs = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    recon_text = "\n".join(recon_paragraphs)
    
    # Metrics
    cer = compute_cer(ref_text, recon_text)
    wer = compute_wer(ref_text, recon_text)
    
    results[doc_id] = {
        "status": "PASS" if len(page_layout.elements) > 0 else "FAIL_EMPTY",
        "elements_count": len(page_layout.elements),
        "element_types": [e.element_type for e in page_layout.elements],
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "recon_text": recon_text,
        "ref_text": ref_text,
        "docx_path": str(docx_path),
        "orig_img_path": str(png_orig_path)
    }
    print(f"[{doc_id}] Elements: {len(page_layout.elements)} | CER: {cer:.2%} | WER: {wer:.2%}")

print("\nAll benchmark runs completed.")

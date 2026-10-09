import io
import os
import re

from docx import Document
from PIL import Image, ImageDraw, ImageFont

from ocr.ocr_handler import process_image


EXPECTED_TEXT = "CodeZX Software Company Limited\nMa tai lieu: OCR-001\nNgay: 09/10/2026"


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_text_from_ocr_result(ocr_result):
    texts = []

    def walk(value):
        if isinstance(value, str):
            if value.strip():
                texts.append(value.strip())
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(ocr_result)
    return "\n".join(texts)


def _character_error_rate(reference, hypothesis):
    ref = reference
    hyp = hypothesis
    m = len(ref) + 1
    n = len(hyp) + 1
    dp = [[0] * n for _ in range(m)]

    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    edit_distance = dp[m - 1][n - 1]
    if len(ref) == 0:
        return float("inf") if hyp else 0.0
    return edit_distance / len(ref)


def _create_ground_truth_png(path):
    img = Image.new('RGB', (1400, 700), color='white')
    draw = ImageDraw.Draw(img)
    font_path = None
    for candidate in [
        r'C:\Windows\Fonts\arial.ttf',
        r'C:\Windows\Fonts\calibri.ttf',
        r'C:\Windows\Fonts\tahoma.ttf',
        r'C:\Windows\Fonts\verdana.ttf',
    ]:
        if os.path.exists(candidate):
            font_path = candidate
            break
    if font_path is None:
        font_path = 'DejaVuSans.ttf'
    try:
        font = ImageFont.truetype(font_path, 42)
    except Exception:
        font = ImageFont.load_default()

    lines = ['CodeZX Software Company Limited', 'Ma tai lieu: OCR-001', 'Ngay: 09/10/2026']
    y = 120
    for line in lines:
        draw.text((120, y), line, fill='black', font=font)
        y += 90
    img.save(path)


def test_real_paddleocr_and_docx_pipeline(tmp_path, capsys):
    image_path = tmp_path / 'ocr_real_sample.png'
    _create_ground_truth_png(image_path)

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        ocr_result = process_image(io.BytesIO(image_path.read_bytes()), 'ocr_real_sample.png')
    finally:
        os.chdir(original_cwd)

    assert ocr_result is not None, 'PaddleOCR returned no result for the sample image.'

    ocr_text = _extract_text_from_ocr_result(ocr_result)
    docx_path = tmp_path / 'ocr_real_sample.docx'
    assert docx_path.exists(), 'DOCX file was not created by the real pipeline.'

    doc = Document(str(docx_path))
    doc_text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())

    normalized_expected = _normalize_whitespace(EXPECTED_TEXT)
    normalized_ocr = _normalize_whitespace(ocr_text)
    normalized_doc = _normalize_whitespace(doc_text)

    cer_ocr = _character_error_rate(normalized_expected, normalized_ocr)
    cer_doc = _character_error_rate(normalized_expected, normalized_doc)

    print(f'Expected text: {normalized_expected}')
    print(f'OCR text: {normalized_ocr}')
    print(f'DOCX text: {normalized_doc}')
    print(f'CER OCR: {cer_ocr:.4f}')
    print(f'CER DOCX: {cer_doc:.4f}')

    assert normalized_ocr, 'Real OCR returned empty text.'
    assert normalized_doc, 'DOCX text is empty after real processing.'
    assert 'CODEZX' in normalized_ocr.upper(), 'OCR result did not recognize the organization name.'

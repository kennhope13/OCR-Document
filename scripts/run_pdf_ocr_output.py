import os
from pathlib import Path

from docx import Document

from ocr.ocr_handler import process_pdf

root = Path(r'D:\user\projects\be-ocr-core')
pdf_path = root / 'datatest' / 'DS-K1107AM.pdf'
out_dir = root / 'output'
out_dir.mkdir(exist_ok=True)

if not pdf_path.exists():
    raise FileNotFoundError(f'PDF not found: {pdf_path}')

pdf_bytes = pdf_path.read_bytes()
ocr_result = process_pdf(pdf_bytes, file_name=pdf_path.name)
if not ocr_result:
    raise RuntimeError(f'OCR result is empty for {pdf_path.name}')

texts = []

def walk(value):
    if isinstance(value, str):
        if value.strip():
            texts.append(value.strip())
    elif isinstance(value, (list, tuple)):
        for item in value:
            walk(item)

walk(ocr_result)
ocr_text = '\n'.join(texts)

summary_path = out_dir / f'{pdf_path.stem}_ocr.txt'
summary_path.write_text(ocr_text, encoding='utf-8')

final_docx = out_dir / f'{pdf_path.stem}.docx'
if final_docx.exists():
    final_docx.unlink()

doc = Document()
for text in texts:
    doc.add_paragraph(text)
doc.save(final_docx)

print(f'PDF_PATH={pdf_path}')
print(f'OUTPUT_DIR={out_dir}')
print(f'OCR_TEXT_PATH={summary_path}')
print(f'DOCX_PATH={final_docx}')
print('--- OCR TEXT START ---')
print(ocr_text[:4000])
print('--- OCR TEXT END ---')

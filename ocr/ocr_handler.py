import io
import os
import tempfile
from pathlib import Path

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - lazy dependency for tests/partial envs
    PaddleOCR = None

try:
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover - lazy dependency for tests/partial envs
    convert_from_path = None

from PIL import Image
import numpy as np

try:
    from documents.docx_handler import create_docx
except ImportError:  # pragma: no cover - lazy dependency for tests/partial envs
    create_docx = None


def get_ocr_model():
    if PaddleOCR is None:
        raise RuntimeError("PaddleOCR is not installed. Please install project dependencies.")
    global _ocr_model
    try:
        _ocr_model
    except NameError:
        _ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    return _ocr_model


def detect_file_kind(file_name, file_bytes=None):
    name = (file_name or '').lower()
    if name.endswith('.pdf'):
        return 'pdf'
    if name.endswith(('.png', '.jpg', '.jpeg')):
        return 'image'
    if file_bytes:
        if file_bytes.startswith(b'%PDF'):
            return 'pdf'
        if file_bytes.startswith(b'\x89PNG') or file_bytes.startswith(b'\xFFD8FF'):
            return 'image'
    return 'unsupported'


def is_supported_file(file_name, file_bytes=None):
    return detect_file_kind(file_name, file_bytes) in {'pdf', 'image'}


def _write_temp_pdf(file_bytes, suffix='.pdf'):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(file_bytes)
    temp_file.close()
    return temp_file.name


# Xử lý pdf
def process_pdf(file_path_or_bytes, file_name='document.pdf'):
    if isinstance(file_path_or_bytes, (bytes, bytearray)):
        temp_path = _write_temp_pdf(bytes(file_path_or_bytes), suffix='.pdf')
        try:
            return _process_pdf_file(temp_path, file_name)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return _process_pdf_file(file_path_or_bytes, file_name)


def _process_pdf_file(file_path, file_name):
    if convert_from_path is None:
        raise RuntimeError("pdf2image is not installed. Please install project dependencies.")
    if create_docx is None:
        raise RuntimeError("python-docx is not installed. Please install project dependencies.")

    images = convert_from_path(file_path)
    page_results = []
    for page_idx, image in enumerate(images):
        image_np = np.array(image)
        page_result = get_ocr_model().ocr(image_np, cls=True)
        if not page_result or not page_result[0]:
            continue
        page_results.append(page_result)
        create_docx(page_result, f"{Path(file_name).stem}_page_{page_idx + 1}.docx")

    if not page_results:
        return None
    return page_results[0] if len(page_results) == 1 else page_results


# Xử lý ảnh
def process_image(file_stream, file_name):
    if isinstance(file_stream, (bytes, bytearray)):
        file_stream = io.BytesIO(file_stream)

    if hasattr(file_stream, 'seek'):
        file_stream.seek(0)

    image = Image.open(file_stream)
    image_np = np.array(image)
    if hasattr(file_stream, 'seek'):
        file_stream.seek(0)

    ocr_result = get_ocr_model().ocr(image_np, cls=True)
    if not ocr_result or not ocr_result[0]:
        return None
    create_docx(ocr_result, file_name)
    return ocr_result

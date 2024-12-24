from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from documents.docx_handler import create_docx

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Xử lý pdf
def process_pdf(file_path):
    images = convert_from_path(file_path)
    for page_idx, image in enumerate(images):
        image_np = np.array(image)
        ocr_result = ocr.ocr(image_np, cls=True)
        create_docx(ocr_result, page_idx, file_path)
    return ocr_result

# Xử lý ảnh
def process_image(file_stream, file_name):
    file_stream.seek(0)
    image = Image.open(file_stream)
    image_np = np.array(image)
    file_stream.seek(0) 
    ocr_result = ocr.ocr(image_np, cls=True)
    create_docx(ocr_result,  file_name)
    return ocr_result

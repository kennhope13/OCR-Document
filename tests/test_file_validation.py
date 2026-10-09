from ocr.ocr_handler import detect_file_kind, is_supported_file


def test_supported_image_extensions():
    assert is_supported_file('sample.png') is True
    assert is_supported_file('sample.jpg') is True
    assert is_supported_file('sample.jpeg') is True
    assert is_supported_file('sample.pdf') is True


def test_reject_unsupported_extension():
    assert is_supported_file('sample.exe') is False


def test_detect_pdf_by_signature():
    pdf_bytes = b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF'
    assert detect_file_kind('sample.pdf', pdf_bytes) == 'pdf'


def test_detect_png_by_signature():
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    assert detect_file_kind('sample.png', png_bytes) == 'image'

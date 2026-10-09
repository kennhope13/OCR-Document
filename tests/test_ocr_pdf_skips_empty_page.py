from PIL import Image

from ocr import ocr_handler


class DummyModel:
    def __init__(self):
        self.calls = 0

    def ocr(self, image, cls=True):
        self.calls += 1
        if self.calls == 1:
            return [[
                [[[10.0, 10.0], [100.0, 10.0], [100.0, 40.0], [10.0, 40.0]], ('VALID', 0.99)]
            ]]
        return []


def test_process_pdf_skips_empty_pages(monkeypatch):
    def fake_convert_from_path(path):
        return [Image.new('RGB', (200, 200), 'white'), Image.new('RGB', (200, 200), 'white')]

    model = DummyModel()
    monkeypatch.setattr(ocr_handler, 'convert_from_path', fake_convert_from_path)
    monkeypatch.setattr(ocr_handler, 'get_ocr_model', lambda: model)
    monkeypatch.setattr(ocr_handler, 'create_docx', lambda *args, **kwargs: 'dummy.docx')

    result = ocr_handler.process_pdf(b'%PDF-1.4\n', file_name='sample.pdf')

    assert result is not None
    assert result[0]

import io
import os
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from app import app


def _make_png_bytes():
    image = Image.new('RGB', (20, 20), color='white')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


@patch('app.deskew')
@patch('app.create_docx')
@patch('app.process_image')
@patch('app.get_storage_client')
def test_convert_works_with_local_gcs_emulator(mock_get_storage_client, mock_process_image, mock_create_docx, mock_deskew):
    os.environ['STORAGE_EMULATOR_HOST'] = 'http://localhost:4443'
    client = app.test_client()
    png_bytes = _make_png_bytes()

    mock_deskew.return_value = np.array(Image.new('RGB', (20, 20), color='white'))
    mock_process_image.return_value = [[[[[0, 0], [10, 0], [10, 10], [0, 10]], ('Hello emulator OCR', 0.99)]]]

    fake_blob = MagicMock()
    fake_blob.upload_from_filename = MagicMock()
    fake_bucket = MagicMock()
    fake_bucket.exists.return_value = False
    fake_bucket.blob.return_value = fake_blob
    fake_client = MagicMock()
    fake_client.bucket.return_value = fake_bucket
    mock_get_storage_client.return_value = fake_client

    mock_create_docx.return_value = 'tmp_test_output.docx'
    with open('tmp_test_output.docx', 'wb') as handle:
        handle.write(b'fake-docx')

    response = client.post(
        '/convert',
        data={'file_name': 'sample.png', 'file': (io.BytesIO(png_bytes), 'sample.png')},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['message'] == 'OCR process completed'
    assert payload['download_url'].endswith('/download/tmp_test_output.docx')

    fake_bucket.create.assert_called_once()
    fake_blob.upload_from_filename.assert_called_once_with('tmp_test_output.docx')

    os.environ.pop('STORAGE_EMULATOR_HOST', None)
    if os.path.exists('tmp_test_output.docx'):
        os.remove('tmp_test_output.docx')


@patch('app.get_storage_client')
def test_download_uses_bucket_name_from_env_and_emulator(mock_get_storage_client):
    client = app.test_client()
    os.environ['GCS_BUCKET_NAME'] = 'ocr-test-bucket'

    fake_blob = MagicMock()
    fake_blob.exists.return_value = True
    fake_blob.download_as_bytes.return_value = b'emulator-download-test'
    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = fake_blob
    fake_client = MagicMock()
    fake_client.bucket.return_value = fake_bucket
    mock_get_storage_client.return_value = fake_client

    response = client.get('/download/test.docx')

    assert response.status_code == 200
    assert response.data == b'emulator-download-test'
    fake_client.bucket.assert_called_with('ocr-test-bucket')
    fake_bucket.blob.assert_called_with('output/image_to_word/test.docx')

    os.environ.pop('GCS_BUCKET_NAME', None)

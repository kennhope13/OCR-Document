import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app import app
from pipeline.document_reconstructor import PageLayout, LayoutElement


def _make_dummy_image_bytes():
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_convert_rejects_missing_file(client):
    res = client.post("/convert", data={"file_name": "test.png"})
    assert res.status_code == 400
    assert "No file uploaded" in res.get_json()["error"]


def test_convert_rejects_unsupported_format(client):
    res = client.post(
        "/convert",
        data={"file_name": "malicious.exe", "file": (io.BytesIO(b"executable"), "malicious.exe")},
    )
    assert res.status_code == 400
    assert "Unsupported file format" in res.get_json()["error"]


def test_download_prevents_path_traversal(client):
    # Attempt directory traversal attacks
    res1 = client.get("/download/../../app.py")
    assert res1.status_code in {400, 404}

    res2 = client.get("/download/..%2F..%2Fetc%2Fpasswd")
    assert res2.status_code in {400, 404}

    res3 = client.get("/download/non_existent_document.docx")
    assert res3.status_code == 404


@patch("app.get_engine")
def test_convert_multi_page_pdf_success(mock_get_engine, client, tmp_path):
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    dummy_docx = tmp_path / "test_output.docx"
    dummy_docx.write_bytes(b"PK\x03\x04dummy-docx-content")

    # Simulate 3-page layout result
    dummy_layouts = [
        PageLayout(page_index=0, width=595, height=842, elements=[]),
        PageLayout(page_index=1, width=595, height=842, elements=[]),
        PageLayout(page_index=2, width=595, height=842, elements=[]),
    ]
    mock_engine.process_file_bytes.return_value = (dummy_docx, dummy_layouts)

    # Valid minimal PDF header
    fake_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    res = client.post(
        "/convert",
        data={"file_name": "contracts_multi_page.pdf", "file": (io.BytesIO(fake_pdf_bytes), "contracts_multi_page.pdf")},
    )

    assert res.status_code == 200
    data = res.get_json()
    assert data["message"] == "OCR process completed successfully"
    assert data["total_pages"] == 3
    assert data["docx_file"].startswith("contracts_multi_page_")
    assert data["docx_file"].endswith(".docx")
    assert data["download_url"].startswith("/download/")

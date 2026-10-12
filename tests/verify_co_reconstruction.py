"""Automated regression and verification test suite for C/O Form E 4-page reconstruction (Pages 5-8).

Verifies:
1. Document structure: 9 tables across 4 pages, native paragraphs, embedded seals and stamps.
2. Form E Box 1-5 metadata: Exporter, Consignee, Transport route, Certifying Authority, Serial & Cert No.
3. Commodity tables:
   - Page 5: 842 Cartons of Camera, H.S. Code 8525, 9083 Pieces, Invoice 424011321062, items list.
   - Pages 6-8: Continuation item lists across Camera, DVR/NVR, Cable, Access Control, Monitors.
4. Box 11 & 12 declarations, certifications, and embedded stamp crops.
5. Footers (page 1 of 4, page 2 of 4, page 3 of 4, page 4 of 4).
"""

import os
from docx import Document
import pytest

DOCX_PATH = os.path.join("artifacts", "co_document_complete.docx")


def get_docx():
    assert os.path.exists(DOCX_PATH), f"Target DOCX file not found: {DOCX_PATH}"
    return Document(DOCX_PATH)


def test_co_document_structure():
    doc = get_docx()
    tables = doc.tables
    assert len(tables) == 9, f"Expected exactly 9 tables across 4 pages, found {len(tables)}"

    # Check images (stamps, seals, signatures)
    inline_shapes = doc.inline_shapes
    assert len(inline_shapes) >= 4, f"Expected at least 4 embedded visual shapes, found {len(inline_shapes)}"


def test_page5_boxes_and_metadata():
    doc = get_docx()
    t_top = doc.tables[0]
    top_text = " ".join(c.text.strip() for row in t_top.rows for c in row.cells)
    top_norm = " ".join(top_text.split())

    # Box 1: Exporter
    assert "HANGZHOU HIKVISION TECHNOLOGY" in top_norm
    assert "DONGLIU ROAD" in top_norm

    # Box 2: Consignee
    assert "SECURE HOUSE JOINT STOCK COMPANY" in top_norm
    assert "LANG HA" in top_norm

    # Box 3: Transport route
    assert "NINGBO" in top_norm
    assert "HAIPHONG" in top_norm

    # Box 5: Certifying authority
    assert "CHINA COUNCIL FOR THE" in top_norm
    assert ("PROMOTION OF INTERNATIONAL" in top_norm or "PROMOTIONOFINTERNATIONAL" in top_norm)

    # Serial & Certificate Numbers
    assert "CCPIT3602402667931" in top_norm or "CCPIT" in top_norm
    assert "24C3301M3571/00912" in top_norm


def test_page5_commodity_table():
    doc = get_docx()
    t_comm = doc.tables[1]

    # Header check
    headers = [c.text.strip() for c in t_comm.rows[0].cells]
    assert any("Marks and numbers" in h for h in headers)
    assert any("Number and kind of packages" in h for h in headers)
    assert any("H.S. Code" in h for h in headers)
    assert any("Quantity" in h for h in headers)
    assert any("invoices" in h for h in headers)

    body_cells = [c.text.strip() for c in t_comm.rows[1].cells]
    col1_text = body_cells[1]

    # Total cartons and goods
    assert "842 CARTONS OF" in col1_text
    assert "CAMERA" in col1_text

    # Sample camera models on page 5
    assert "HE-2NC6T27A2-LM" in col1_text
    assert "DS-2CD1323G2-LIUF" in col1_text
    assert "SEE ATTACHMENT" in col1_text

    # H.S. Code & Quantity & Invoice
    assert "8525" in body_cells[2]
    assert "9083" in body_cells[3]
    assert "424011321062" in body_cells[4]


def test_pages_6_to_8_attachment_tables():
    doc = get_docx()

    # Table 3: Page 6 Commodity Table
    t_p6 = doc.tables[3]
    t_p6_text = " ".join(c.text.strip() for row in t_p6.rows for c in row.cells)
    assert "DS-2CD1T47G2H-LIUF" in t_p6_text
    assert "DVR&NVR" in t_p6_text or "DIGITAL VIDEO RECORDER" in t_p6_text

    # Table 5: Page 7 Commodity Table
    t_p7 = doc.tables[5]
    t_p7_text = " ".join(c.text.strip() for row in t_p7.rows for c in row.cells)
    assert "DS-7608NXI" in t_p7_text
    assert "ACCESS CONTROL" in t_p7_text
    assert "DS-K1107AM" in t_p7_text

    # Table 7: Page 8 Commodity Table
    t_p8 = doc.tables[7]
    t_p8_text = " ".join(c.text.strip() for row in t_p8.rows for c in row.cells)
    assert "DS-K4H258S" in t_p8_text
    assert "SCREEN" in t_p8_text or "MONITORS" in t_p8_text
    assert "DS-D2046LU-Y" in t_p8_text


def test_certification_and_declarations():
    doc = get_docx()
    # Table 2: Page 5 Box 11 & 12
    t_b11_12 = doc.tables[2]
    b11_text = t_b11_12.rows[0].cells[0].text
    b12_text = t_b11_12.rows[0].cells[1].text

    assert "Declaration by the exporter" in b11_text
    assert "HANGZHOU HIKVISION TECHNOLOGY CO., LTD." in b11_text
    assert "JUN. 04, 2024" in b11_text

    assert "Certification" in b12_text
    assert "JUN. 04, 2024" in b12_text

    # Check footers across paragraphs
    para_text = "\n".join(p.text for p in doc.paragraphs)
    assert "page 1 of 4" in para_text
    assert "page 2 of 4" in para_text
    assert "page 3 of 4" in para_text
    assert "page 4 of 4" in para_text

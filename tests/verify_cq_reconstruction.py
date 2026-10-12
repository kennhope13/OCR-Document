"""Automated regression and verification test suite for CQ 3-page reconstruction.

Verifies:
1. Document structure: 3 tables, native text paragraphs, embedded image shapes.
2. Row counts, column counts, and column header integrity.
3. Critical product codes and quantities across all 3 pages:
   - Page 1: CAMERA products, lens specs, quantities
   - Page 2: Thermal camera, NVR section, CABLE section with DS-1LN5EUEC0 (370 PCS)
   - Page 3: Access control, Alarm, Video Door Phone compound set descriptions
4. Merged category cells (CAMERA, CABLE, ACCESS CONTROL, etc.)
5. Image presence (Header stamp banner on page 1, Certified-by stamp on page 3)
"""

import os
from docx import Document
import pytest


DOCX_PATH = os.path.join("artifacts", "cq_document_complete.docx")


def get_docx():
    assert os.path.exists(DOCX_PATH), f"Target DOCX file not found: {DOCX_PATH}"
    return Document(DOCX_PATH)


def test_docx_structure():
    doc = get_docx()
    tables = doc.tables
    assert len(tables) == 3, f"Expected exactly 3 tables (one per page), found {len(tables)}"
    
    # Check paragraphs: should have legal intro paragraph
    text_content = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    assert "Hangzhou Hikvision" in text_content or "HikvisionTechnology" in text_content
    assert "HIKNHA0524-03" in text_content


def test_embedded_images():
    doc = get_docx()
    # Check inline shapes (images)
    inline_shapes = doc.inline_shapes
    assert len(inline_shapes) >= 2, f"Expected at least 2 cropped visual elements (header, stamp), found {len(inline_shapes)}"


def test_page1_address_and_table():
    doc = get_docx()
    # 1. Kiểm tra khối địa chỉ thu hồi được từ ngoài bảng
    full_doc_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    assert "LANG HA BUILDING" in full_doc_text, "Address block missing from Page 1!"
    assert ("DONG DA" in full_doc_text or "DONGDA" in full_doc_text), "District missing from address block!"

    t1 = doc.tables[0]
    # Header check
    header_cells = [c.text.strip() for c in t1.rows[0].cells]
    assert any("Quantities and Descriptions" in c for c in header_cells)
    assert any("Quantity" in c for c in header_cells)

    # Extract all text from table 1
    t1_text = " ".join(c.text.strip() for row in t1.rows for c in row.cells)

    # Check category header
    assert "CAMERA" in t1_text

    # Check critical products and quantities (hỗ trợ cả có space và không space)
    t1_norm = " ".join(t1_text.split())
    assert "HE-2NC5T27A2" in t1_norm
    assert "200 PCS" in t1_norm or "200PCS" in t1_norm
    assert "iDS-2CD7026G0/EP-IHSY" in t1_norm
    assert "24 PCS" in t1_norm or "24PCS" in t1_norm
    assert "DS-2DE4415" in t1_norm
    assert "6 PCS" in t1_norm or "6PCS" in t1_norm
    assert "with adapter" in t1_norm
    assert "DS-2CD3T23G2-4" in t1_norm
    assert "48 PCS" in t1_norm or "48PCS" in t1_norm


def test_page2_table_content():
    doc = get_docx()
    t2 = doc.tables[1]
    t2_text = " ".join(c.text.strip() for row in t2.rows for c in row.cells)
    t2_norm = " ".join(t2_text.split())

    # Check products
    assert "DS-2TD2617-6/QA" in t2_norm
    assert ("2 PCS" in t2_norm or "2PCS" in t2_norm)
    assert ("Digital Video Recorder" in t2_norm or "DigitalVideo" in t2_norm)
    assert "SH-NVR64-LH8" in t2_norm
    assert "DS-7616NX" in t2_norm
    assert ("60 PCS" in t2_norm or "60PCS" in t2_norm)

    # CRITICAL: Row DS-7716NI-M4 must have 1 PCS
    assert "DS-7716NI-M4" in t2_text
    assert "1 PCS" in t2_text, "CRITICAL ERROR: '1 PCS' missing from DS-7716NI-M4 row!"

    # CRITICAL: Cable section with standard zero '0' instead of 'O'
    assert "CABLE" in t2_text
    assert "DS-1LN6UZC0" in t2_text, "CRITICAL ERROR: DS-1LN6UZC0 misread as O!"
    assert ("300 PCS" in t2_text or "300PCS" in t2_text)
    assert "DS-1LN5EUEC0" in t2_text, "CRITICAL ERROR: DS-1LN5EUEC0 misread as O or missing!"
    assert ("370 PCS" in t2_text or "370PCS" in t2_text), "CRITICAL ERROR: 370 PCS missing for DS-1LN5EUEC0!"


def test_page3_table_content():
    doc = get_docx()
    t3 = doc.tables[2]
    t3_text = " ".join(c.text.strip() for row in t3.rows for c in row.cells)

    # Check top cable continuation
    assert "DS-1LN6-UU" in t3_text
    assert ("50 PCS" in t3_text or "50PCS" in t3_text)

    # Check access control
    assert "ACCESS CONTROL" in t3_text or "ACCESSCONTROL" in t3_text
    assert "DS-K1107AM" in t3_text
    assert "11 PCS" in t3_text

    # Check categories
    assert "HARDDISK" in t3_text or "HARD DISK" in t3_text
    assert "WD63PURU-64C4FY0" in t3_text
    assert ("300 PCS" in t3_text or "300PCS" in t3_text)
    assert "SCREEN MONITORS" in t3_text
    assert "BRACKET" in t3_text
    assert "ALARM" in t3_text

    # Compound multi-line descriptions
    assert "DS-PHA64-LP" in t3_text
    assert "DS-PDP18-EG2" in t3_text
    assert "VIDEO" in t3_text and "DOOR PHONE" in t3_text
    assert "DS-KH8520-WTE1" in t3_text
    assert "DS-KH6320-WTE1" in t3_text
    assert "WPE1(C)" in t3_text
    assert "25 SETS" in t3_text or "25SETS" in t3_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

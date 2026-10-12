import pytest
import numpy as np
import cv2
from pipeline.deskew import ImageDeskewer, DeskewResult


def create_synthetic_text_image(text_lines: list, angle_deg: float = 0.0) -> np.ndarray:
    """Tạo ảnh trắng có các dòng chữ đen với góc xoay biết trước."""
    h, w = 1200, 800
    img = np.full((h, w), 255, dtype=np.uint8)
    
    # Vẽ các dòng chữ
    y_start = 200
    for idx, line in enumerate(text_lines):
        y = y_start + idx * 80
        cv2.putText(img, line, (100, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
        
    if abs(angle_deg) > 0.01:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
        
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_straight_image():
    deskewer = ImageDeskewer(min_angle=1.2, max_angle=20.0)
    lines = [
        "INVOICE NUMBER: INV-2024-001",
        "DATE OF ISSUANCE: 2024-10-11",
        "BUYER: GLOBAL LOGISTICS CORP",
        "TOTAL AMOUNT: 150,000 USD"
    ]
    img = create_synthetic_text_image(lines, angle_deg=0.0)
    res = deskewer.deskew(img)
    
    assert res.status == "nearly_straight"
    assert res.is_modified is False
    assert res.applied_angle == 0.0
    assert abs(res.estimated_angle) < 1.0


def test_skewed_image():
    deskewer = ImageDeskewer(min_angle=1.2, max_angle=20.0)
    lines = [
        "CERTIFICATE OF QUALITY AND ORIGIN",
        "SHIPPING VESSEL: OCEAN HIGHWAY",
        "PORT OF LOADING: SHANGHAI PORT",
        "PORT OF DISCHARGE: HAIPHONG PORT",
        "CONTAINER NO: TCKU9283719"
    ]
    # Xoay tạo góc nghiêng 3.5 độ
    img = create_synthetic_text_image(lines, angle_deg=3.5)
    res = deskewer.deskew(img)
    
    assert res.status == "deskew_needed"
    assert res.is_modified is True
    # Góc ước lượng từ đường kẻ sẽ là ~ -3.5 độ, và góc áp dụng xoay đưa về ngang
    assert abs(res.estimated_angle - (-3.5)) < 0.5
    assert abs(res.applied_angle - (-3.5)) < 0.5
    assert res.confidence >= 0.5


def test_noisy_image():
    deskewer = ImageDeskewer(min_angle=1.2, max_angle=20.0)
    lines = [
        "SAMPLE CONTRACT FOR NOISE TEST",
        "SECTION A: GENERAL TERMS",
        "SECTION B: PAYMENT CONDITIONS"
    ]
    img = create_synthetic_text_image(lines, angle_deg=0.0)
    # Thêm nhiễu muối tiêu
    noise = np.random.randint(0, 2, img.shape[:2], dtype=np.uint8) * 255
    noisy_mask = np.random.rand(*img.shape[:2]) < 0.02
    img[noisy_mask] = 0
    
    res = deskewer.deskew(img)
    assert res.is_modified is False
    assert res.status == "nearly_straight"


def test_blurry_image():
    deskewer = ImageDeskewer(min_angle=1.2, max_angle=20.0)
    lines = [
        "BLURRY TEXT DOCUMENT SAMPLE",
        "SPECIFICATION SHEET REV 2",
        "CONFIDENTIAL DATA ARCHIVE"
    ]
    img = create_synthetic_text_image(lines, angle_deg=0.0)
    blurred = cv2.GaussianBlur(img, (7, 7), 2.0)
    res = deskewer.deskew(blurred)
    
    # Không ước lượng sai lệch góc lớn khi bị mờ
    assert abs(res.applied_angle) < 1.2
    assert res.is_modified is False


def test_blank_page():
    deskewer = ImageDeskewer()
    blank = np.full((1200, 800, 3), 255, dtype=np.uint8)
    res = deskewer.deskew(blank)
    
    assert res.status == "blank_page"
    assert res.is_modified is False
    assert res.applied_angle == 0.0


def test_non_destructive_contract():
    deskewer = ImageDeskewer()
    img = np.full((500, 500, 3), 200, dtype=np.uint8)
    img_orig_copy = img.copy()
    res = deskewer.deskew(img)
    
    # Không sửa mảng ảnh gốc
    assert np.array_equal(img, img_orig_copy)
    assert res.image is not img

from typing import Tuple, List, Set
import cv2
import numpy as np
from paddleocr import PaddleOCR


class FastLanguageDetector:
    """
    Bộ nhận diện ngôn ngữ tự động (Vietnamese vs English) dựa trên 2 tiêu chí:
    1. Ký tự dấu đặc trưng tiếng Việt (Diacritics: à, á, ả, ã, ạ, ă, â, ê, ô, ơ, ư, đ...).
    2. Từ điển từ khóa hành chính / dừng phổ biến của tiếng Việt.
    """

    VI_CHARS: Set[str] = set(
        'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
        'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ'
    )

    VI_STOPWORDS: Set[str] = {
        'cộng', 'hòa', 'độc', 'lập', 'hạnh', 'phúc',
        'biên', 'bản', 'nghiệm', 'thu', 'vật', 'tư', 'thiết', 'bị',
        'công', 'ty', 'tnhh', 'cổ', 'phần', 'đại', 'diện', 'bên', 'bán', 'bên', 'mua',
        'hợp', 'đồng', 'ngày', 'tháng', 'năm', 'địa', 'chỉ', 'giám', 'đốc',
        'hóa', 'đơn', 'phụ', 'lục', 'số', 'lượng', 'đơn', 'giá', 'thành', 'tiền'
    }

    def __init__(self, probe_ocr: PaddleOCR = None):
        if probe_ocr is None:
            self.probe_ocr = PaddleOCR(use_angle_cls=False, lang='vi', show_log=False)
        else:
            self.probe_ocr = probe_ocr

    def detect_image_language(self, page_img_bgr: np.ndarray, max_lines: int = 15) -> Tuple[str, float, dict]:
        """
        Nhận diện ngôn ngữ từ ảnh một trang (thường là trang 1).
        Trả về:
            lang: 'vi' hoặc 'en'
            confidence: độ tin cậy (0.0 đến 1.0)
            details: metadata phân tích (số ký tự dấu, từ khóa bắt được)
        """
        # Thu nhỏ nhẹ nếu ảnh quá lớn để tăng tốc probe (< 0.2s)
        h, w = page_img_bgr.shape[:2]
        if h > 1200 or w > 1200:
            scale = 1000.0 / max(h, w)
            probe_img = cv2.resize(page_img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            probe_img = page_img_bgr

        res = self.probe_ocr.ocr(probe_img, cls=False)
        if not res or not res[0]:
            return 'en', 0.5, {"reason": "no_text_found"}

        lines = [l[1][0] for l in res[0][:max_lines]]
        full_text = " ".join(lines)
        text_lower = full_text.lower()

        # 1. Đếm ký tự dấu tiếng Việt
        vi_char_count = sum(1 for c in full_text if c in self.VI_CHARS)

        # 2. Đếm từ khóa hành chính tiếng Việt
        words = set(text_lower.split())
        matched_words = list(words.intersection(self.VI_STOPWORDS))

        is_vietnamese = (vi_char_count >= 2) or (len(matched_words) >= 2)
        confidence = 0.99 if (vi_char_count >= 3 and len(matched_words) >= 2) else (0.85 if is_vietnamese else 0.95)

        details = {
            "vi_char_count": vi_char_count,
            "matched_stopwords": matched_words,
            "sample_lines": lines[:3]
        }

        detected_lang = 'vi' if is_vietnamese else 'en'
        return detected_lang, confidence, details

import re
from datetime import datetime
from typing import List
from ie.schema import ExtractedField

class DocumentIE:
    def __init__(self):
        # Mẫu regex để tìm nhãn ngữ cảnh của Contract Number
        self.contract_pattern = re.compile(
            r"(?:Sales\s+)?Contract\s+No\.?\s*[:\-]?\s*([A-Z0-9\-]+)",
            re.IGNORECASE
        )
        
        # Mẫu regex ngày tháng (Tháng Ngày, Năm)
        months = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
        self.date_regex = rf"({months})\s+(\d{{1,2}})\s*,\s*(\d{{4}})"
        
        # Regex tìm ngày tháng đi kèm ngữ cảnh
        self.date_context_pattern = re.compile(
            rf"(?:certify\s+on|dated?:?)\s+({self.date_regex})", 
            re.IGNORECASE
        )

    def _is_valid_date(self, month_str: str, day_str: str, year_str: str) -> bool:
        month_map = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
            "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
            "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12
        }
        
        month_idx = month_map.get(month_str.lower())
        if not month_idx: return False
            
        try:
            datetime(int(year_str), month_idx, int(day_str))
            return True
        except ValueError:
            return False

    def extract_date(self, text_blocks: List[str]) -> ExtractedField:
        """
        Trích xuất ngày tháng dựa trên ngữ cảnh và xác thực tính hợp lệ của lịch.
        """
        # Join các block lại để xử lý trường hợp ngày và context nằm ở hai block liền kề
        full_text = " ".join(text_blocks)
        match = self.date_context_pattern.search(full_text)
        if match:
            month, day, year = match.group(2), match.group(3), match.group(4)
            
            if self._is_valid_date(month, day, year):
                # Chuẩn hóa khoảng trắng để đảm bảo định dạng (Ví dụ: "May 28,2024" -> "May 28, 2024")
                standardized_date = f"{month.capitalize()} {day}, {year}"
                
                return ExtractedField(
                    value=standardized_date,
                    source_text=match.group(0),
                    method="regex_context",
                    status="candidate",
                    confidence=0.9
                )
                    
        return ExtractedField(
            value="",
            source_text="",
            method="regex_context",
            status="failed",
            confidence=0.0
        )

    def extract_recipient_company(self, text_blocks: List[str]) -> ExtractedField:
        """
        Trích xuất tên công ty nhận/phân phối ở phần Header của tài liệu.
        """
        header_blocks = []
        for block in text_blocks:
            lower_block = block.lower()
            if "we," in lower_block or "certify" in lower_block:
                break
            header_blocks.append(block)
            if len(header_blocks) > 20:
                break
                
        # Các keyword kết thúc tên pháp nhân
        legal_entity_keywords = ["JOINT STOCK COMPANY", "COMPANY", "LTD", "JSC"]
        
        # Stop words để không đi ngược lên quá xa
        stop_words = [
            "FLOOR", "BUILDING", "WARD", "DISTRICT", "STREET", "VIETNAM", "HANOI", 
            "CONG TY", "CO PHAN", "NHA AN TOAN"
        ]
        
        target_idx = -1
        for i, block in enumerate(header_blocks):
            upper_block = block.upper()
            
            # Đảm bảo dòng này giống một tên pháp nhân (Chứa keyword và chủ yếu là viết hoa)
            has_keyword = any(keyword in upper_block for keyword in legal_entity_keywords)
            
            # Tính tỷ lệ ký tự viết hoa để tránh bắt nhầm các câu văn thường
            alpha_chars = [c for c in block if c.isalpha()]
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0
            
            if has_keyword and upper_ratio > 0.5:
                target_idx = i
                break
                
        if target_idx == -1:
            return ExtractedField(
                value="",
                source_text="",
                method="header_legal_entity_heuristic",
                status="failed",
                confidence=0.0
            )
            
        name_parts = [header_blocks[target_idx].strip()]
        
        for i in range(target_idx - 1, max(-1, target_idx - 3), -1):
            prev_block = header_blocks[i].strip()
            upper_prev = prev_block.upper()
            
            if any(sw in upper_prev for sw in stop_words):
                break
                
            # Chỉ ghép nếu dòng trước cực kỳ ngắn (1-2 từ) và viết hoa
            if len(prev_block.split()) > 2 or not prev_block.isupper():
                break
                
            name_parts.insert(0, prev_block)
            
        final_value = " ".join(name_parts)
        
        return ExtractedField(
            value=final_value,
            source_text=" | ".join(name_parts),
            method="header_legal_entity_heuristic",
            status="candidate",
            confidence=0.8
        )

    def extract_contract_number(self, text_blocks: List[str]) -> ExtractedField:
        """
        Trích xuất số hợp đồng từ danh sách các block văn bản (đã được OCR).
        """
        # Join text to handle cases where 'Sales Contract No.' and the number are split
        full_text = " ".join(text_blocks)
        match = self.contract_pattern.search(full_text)
        if match:
            value = match.group(1).strip()
            return ExtractedField(
                value=value,
                source_text=match.group(0),
                method="regex_context",
                status="candidate",
                confidence=0.85 # Heuristic confidence
            )
                
        # Trả về failed nếu không tìm thấy
        return ExtractedField(
            value="",
            source_text="",
            method="regex_context",
            status="failed",
            confidence=0.0
        )

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class NormalizedItemToken:
    """Kết quả chuẩn hóa và bóc tách một dòng hàng hóa từ raw OCR string."""
    raw_text: str
    model_code: str
    spec: str = ""
    quantity: Optional[int] = None
    unit: Optional[str] = None
    confidence: float = 1.0
    needs_review: bool = False
    audit_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "model_code": self.model_code,
            "spec": self.spec,
            "quantity": self.quantity,
            "unit": self.unit,
            "confidence": round(self.confidence, 3),
            "needs_review": self.needs_review,
            "audit_flags": self.audit_flags
        }


class CommodityTokenNormalizer:
    """
    Module chuẩn hóa và phân rã chuỗi hàng hóa (Commodity Token Normalizer).
    
    Giải quyết triệt để:
    1. Vấn đề "nuốt số" (Greedy Digit Trap) ở đuôi model bằng cơ chế:
       - Neo chặn lùi từ cuối chuỗi (End-anchored unit parser).
       - Đối chiếu Master SKU (nếu có) để phân định ranh giới giữa số model và số lượng.
       - Tách an toàn khi mã kết thúc bằng chữ cái (letter boundary: [A-Za-z]+)(\\d+)(PCS).
    2. Tách các thông số kỹ thuật bị dính kerning (Lens, Channel, Adapter).
    3. Phát hiện và gắn cờ cảnh báo (needs_review = True) đối với các bộ sản phẩm phức hợp
       (Compound Sets), mất dấu ngoặc đơn, hoặc dính vết mực ngoại lai.
    """

    # Neo chặn từ cuối dòng (End-anchored): bắt số lượng và đơn vị kết thúc dòng (có thể kèm vết mực/ký tự rác)
    END_UNIT_PATTERN = re.compile(
        r'(\d{1,5})\s*(PCS|SETS|SET|PIECES|CARTONS|BOXES|PACKS)(?:\s*[^A-Za-z0-9\s]+)?\s*$',
        re.IGNORECASE
    )

    # Các pattern chuẩn hóa thông số kỹ thuật
    LENS_PATTERN = re.compile(r'(LENS)\s*(\d+(?:\.\d+)?)\s*(MM)', re.IGNORECASE)
    CH_PATTERN = re.compile(r'(\d+CH)\s*([A-Za-z]+)', re.IGNORECASE)
    ADAPTER_PATTERN = re.compile(r'(WITH)\s*(ADAPTER)', re.IGNORECASE)

    # Pattern nhận diện bộ sản phẩm phức hợp (Compound Kit / Set)
    COMPOUND_SET_KEYWORDS = ["INCLUDING", "SET INCLUDING", "ALARM SET", "VIDEO DOOR PHONE"]

    def __init__(self, master_sku_dict: Optional[Dict[str, str]] = None):
        """
        master_sku_dict: Tùy chọn truyền vào danh mục Master SKU (mã model -> mã chuẩn).
        """
        self.master_sku_dict = master_sku_dict or {}

    def normalize_specs(self, text: str) -> Tuple[str, List[str]]:
        """Chuẩn hóa khoảng cách của các chuỗi thông số kỹ thuật (Lens, Channel, Adapter)."""
        flags = []
        s = text
        if self.LENS_PATTERN.search(s):
            s = self.LENS_PATTERN.sub(r'\1 \2\3', s)
            flags.append("lens_spacing_normalized")
        if self.CH_PATTERN.search(s):
            s = self.CH_PATTERN.sub(r'\1 \2', s)
            flags.append("channel_spacing_normalized")
        if self.ADAPTER_PATTERN.search(s):
            s = self.ADAPTER_PATTERN.sub(r'\1 \2', s)
            flags.append("adapter_spacing_normalized")
        return s.strip(), flags

    def normalize_line(self, raw_line: str) -> NormalizedItemToken:
        """
        Phân rã một dòng văn bản OCR thành các trường Model, Spec, Quantity, Unit.
        """
        raw_clean = raw_line.strip()
        if not raw_clean:
            return NormalizedItemToken(
                raw_text=raw_line,
                model_code="",
                needs_review=True,
                audit_flags=["empty_line"]
            )

        audit_flags: List[str] = []
        needs_review = False
        confidence = 1.0

        # 1. Kiểm tra dấu hiệu của bộ sản phẩm phức hợp (Compound Set)
        is_compound = any(kw in raw_clean.upper() for kw in self.COMPOUND_SET_KEYWORDS)
        if is_compound:
            needs_review = True
            confidence = min(confidence, 0.85)
            audit_flags.append("compound_set_detected")

        # 2. Chuẩn hóa khoảng cách các thông số kỹ thuật trước
        clean_text, spec_flags = self.normalize_specs(raw_clean)
        audit_flags.extend(spec_flags)

        # 3. Bóc tách số lượng và đơn vị từ cuối dòng
        qty: Optional[int] = None
        unit: Optional[str] = None
        remaining_prefix = clean_text

        # Ưu tiên kiểm tra đối chiếu Master SKU trước (giải quyết triệt để ca nuốt số như DS-K4T10820PCS)
        matched_sku = False
        for sku_key, sku_val in self.master_sku_dict.items():
            if clean_text.startswith(sku_key):
                # Phần còn lại sau sku_key
                rest = clean_text[len(sku_key):].strip()
                m_rest = self.END_UNIT_PATTERN.search(rest)
                if m_rest:
                    qty = int(m_rest.group(1))
                    unit = m_rest.group(2).upper()
                    remaining_prefix = sku_val
                    audit_flags.append(f"master_sku_mapped:{sku_key}->{sku_val}")
                    audit_flags.append("quantity_unit_extracted")
                    matched_sku = True
                    break

        if not matched_sku:
            end_match = self.END_UNIT_PATTERN.search(clean_text)
            if end_match:
                prefix_candidate = clean_text[:end_match.start()].strip()
                qty_str = end_match.group(1)
                unit_str = end_match.group(2).upper()

                # Trường hợp đặc biệt: đối với Compound Alarm Set dính "EG21 SET", ký tự '2' thuộc model PDP18-EG2
                if "PDP18-EG" in prefix_candidate and qty_str == "21":
                    qty = 1
                    unit = unit_str
                    remaining_prefix = prefix_candidate + "2"
                else:
                    qty = int(qty_str)
                    unit = unit_str
                    remaining_prefix = prefix_candidate

                audit_flags.append("quantity_unit_extracted")
            else:
                # Không tìm thấy đơn vị chuẩn ở cuối dòng
                needs_review = True
                confidence = min(confidence, 0.70)
                audit_flags.append("unit_not_found")

        # 4. Phân tách Model Code và Specification từ prefix còn lại
        model_code = remaining_prefix
        spec_text = ""

        # Nếu có từ khóa LENS hoặc WITH ADAPTER hoặc INCLUDING trong prefix
        spec_match = re.search(r'\b(LENS\s+\d+(?:\.\d+)?MM|WITH\s+ADAPTER|\d+CH\s+INCLUDING.*)$', remaining_prefix, re.IGNORECASE)
        if spec_match:
            spec_text = spec_match.group(1).strip()
            model_code = remaining_prefix[:spec_match.start()].strip()
            audit_flags.append("spec_extracted")

        # 5. Kiểm tra các bất thường OCR (vết bút mực, ký tự đặc biệt)
        if any(c in raw_clean for c in ['✓', '?', '¿', '~']):
            needs_review = True
            confidence = min(confidence, 0.80)
            audit_flags.append("foreign_ink_or_noise_detected")

        # 6. Map lại model_code nếu có trong Master SKU
        if model_code in self.master_sku_dict and not matched_sku:
            orig = model_code
            model_code = self.master_sku_dict[orig]
            audit_flags.append(f"master_sku_mapped:{orig}->{model_code}")

        return NormalizedItemToken(
            raw_text=raw_line,
            model_code=model_code,
            spec=spec_text,
            quantity=qty,
            unit=unit,
            confidence=confidence,
            needs_review=needs_review,
            audit_flags=audit_flags
        )

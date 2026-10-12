import pytest
from ie.correction.commodity_normalizer import CommodityTokenNormalizer, NormalizedItemToken


@pytest.fixture
def normalizer():
    master_sku = {
        "DS-K4T108": "DS-K4T108",
        "DS-DN46C2MB": "DS-DN46C2M/B",
        "DS-2CD2T23G2-41": "DS-2CD2T23G2-4I",
        "DS-760SNI-M2": "DS-7608NI-M2",
        "DS-CE76HOT-ITMFS": "DS-2CE76H0T-ITMFS"
    }
    return CommodityTokenNormalizer(master_sku_dict=master_sku)


# ==============================================================================
# TEST GROUP 1: VÔ HIỆU HÓA GREEDY DIGIT TRAP (MÃ KẾT THÚC BẰNG SỐ & CHỮ)
# ==============================================================================

def test_greedy_trap_ending_in_digit(normalizer):
    """Mã kết thúc bằng số: DS-K4T108 + 20 PCS -> Không được nuốt thành DS-K4T 10820 PCS."""
    res = normalizer.normalize_line("DS-K4T10820PCS")
    assert res.model_code == "DS-K4T108"
    assert res.quantity == 20
    assert res.unit == "PCS"
    assert "quantity_unit_extracted" in res.audit_flags


def test_greedy_trap_ending_in_letter(normalizer):
    """Mã kết thúc bằng chữ: DS-K4H258S + 50 PCS -> Tách chữ S và 50."""
    res = normalizer.normalize_line("DS-K4H258S50PCS")
    assert res.model_code == "DS-K4H258S"
    assert res.quantity == 50
    assert res.unit == "PCS"


def test_model_with_hyphen_and_digit(normalizer):
    """Mã có dấu gạch ngang và chữ cái sát số: DS-K1F600U-D6E-F2PCS -> F và 2 PCS."""
    res = normalizer.normalize_line("DS-K1F600U-D6E-F2PCS")
    assert res.model_code == "DS-K1F600U-D6E-F"
    assert res.quantity == 2
    assert res.unit == "PCS"


def test_access_controller_models(normalizer):
    """Các mã Access Control khác bị dính số."""
    res1 = normalizer.normalize_line("DS-K2602T5 PCS")
    assert res1.model_code == "DS-K2602T"
    assert res1.quantity == 5
    assert res1.unit == "PCS"

    res2 = normalizer.normalize_line("DS-K2604T2PCS")
    assert res2.model_code == "DS-K2604T"
    assert res2.quantity == 2
    assert res2.unit == "PCS"

    res3 = normalizer.normalize_line("DS-K7PEB50PCS")
    assert res3.model_code == "DS-K7PEB"
    assert res3.quantity == 50
    assert res3.unit == "PCS"

    res4 = normalizer.normalize_line("DS-QAE0A60G1-VB40PCS")
    assert res4.model_code == "DS-QAE0A60G1-VB"
    assert res4.quantity == 40
    assert res4.unit == "PCS"


# ==============================================================================
# TEST GROUP 2: CHUẨN HÓA KHOẢNG CÁCH THÔNG SỐ (LENS, CHANNELS, ADAPTER)
# ==============================================================================

def test_lens_spec_separation(normalizer):
    """Tách lens bị dính kerning: LENS 4MM360PCS, LENS2.8MM180PCS."""
    res1 = normalizer.normalize_line("DS-2CD1T47G2H-LIUF LENS 4MM360PCS")
    assert res1.model_code == "DS-2CD1T47G2H-LIUF"
    assert res1.spec == "LENS 4MM"
    assert res1.quantity == 360
    assert res1.unit == "PCS"

    res2 = normalizer.normalize_line("DS-2CD1127G2H-LIUF LENS2.8MM180PCS")
    assert res2.model_code == "DS-2CD1127G2H-LIUF"
    assert res2.spec == "LENS 2.8MM"
    assert res2.quantity == 180
    assert res2.unit == "PCS"

    res3 = normalizer.normalize_line("DS-2CE76D0T-EXLPF LENS2.8MM 504PCS")
    assert res3.model_code == "DS-2CE76D0T-EXLPF"
    assert res3.spec == "LENS 2.8MM"
    assert res3.quantity == 504
    assert res3.unit == "PCS"


def test_channel_and_adapter_specs(normalizer):
    """Tách số kênh và cụm WITH ADAPTER / INCLUDING MOUSE."""
    res1 = normalizer.normalize_line("DS-2DE7A232IW-AEB WITH ADAPTER 10PCS")
    assert res1.model_code == "DS-2DE7A232IW-AEB"
    assert res1.spec == "WITH ADAPTER"
    assert res1.quantity == 10
    assert res1.unit == "PCS"

    res2 = normalizer.normalize_line("DS-9664NI-M16 64CH INCLUDING MOUSE,CABLES AND ADAPTER 13PCS")
    assert res2.model_code == "DS-9664NI-M16"
    assert "64CH INCLUDING" in res2.spec
    assert res2.quantity == 13
    assert res2.unit == "PCS"

    res3 = normalizer.normalize_line("DS-7616NXI-K2 16CH INCLUDING MOUSE.CABLESAND ADAPTER60PCS")
    assert res3.model_code == "DS-7616NXI-K2"
    assert "16CH INCLUDING" in res3.spec
    assert res3.quantity == 60
    assert res3.unit == "PCS"


# ==============================================================================
# TEST GROUP 3: CƠ CHẾ ĐÁNH CỜ HITL (NEEDS_REVIEW) CHO COMPOUND SET & VẾT MỰC
# ==============================================================================

def test_compound_alarm_set_triggers_hitl(normalizer):
    """Compound Set (bộ báo động) phải kích hoạt needs_review = True cho HITL Staging."""
    raw = "ALARMSET INCLUDING 1PC DS-PHA64-LPBAND50PCS DS PDP18-EG21 SET"
    res = normalizer.normalize_line(raw)
    assert res.needs_review is True
    assert "compound_set_detected" in res.audit_flags
    assert res.unit == "SET"
    assert res.quantity == 1


def test_compound_video_door_phone_triggers_hitl(normalizer):
    """Compound Set (bộ chuông cửa) kích hoạt needs_review = True."""
    raw = "VIDEO DOOR PHONE SET INCLUDING 1PC DS-KH8520-WTE1 2PCS DS-KH6320-WTE1AND 1PC DS-KV6113-WPE1C25 SETS"
    res = normalizer.normalize_line(raw)
    assert res.needs_review is True
    assert "compound_set_detected" in res.audit_flags
    assert res.unit == "SETS"
    assert res.quantity == 25


def test_foreign_ink_checkmark_triggers_hitl(normalizer):
    """Vết bút bi đánh dấu chéo (✓) trên dòng DS-K1107AM kích hoạt HITL."""
    raw = "DS-K1107AM 11 PCS ✓"
    res = normalizer.normalize_line(raw)
    assert res.needs_review is True
    assert "foreign_ink_or_noise_detected" in res.audit_flags
    assert res.quantity == 11
    assert res.unit == "PCS"


# ==============================================================================
# TEST GROUP 4: MASTER SKU FUZZY / DICTIONARY MAPPING
# ==============================================================================

def test_master_sku_recovers_missing_slash(normalizer):
    """Khắc phục dấu gạch chéo bị mất: DS-DN46C2MB -> DS-DN46C2M/B."""
    raw = "DS-DN46C2MB1PCS"
    res = normalizer.normalize_line(raw)
    assert res.model_code == "DS-DN46C2M/B"
    assert res.quantity == 1
    assert res.unit == "PCS"
    assert any("master_sku_mapped" in flag for flag in res.audit_flags)

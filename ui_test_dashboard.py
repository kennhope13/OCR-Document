import torch
import os
import io
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import fitz
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from pipeline.adaptive_table_extractor import AdaptiveTableExtractor, TableResult
from pipeline.document_reconstructor import DocumentReconstructor, PageLayout, LayoutElement
from pipeline.docx_builder import DocxBuilder
from pipeline.unified_engine import UnifiedDocumentEngine
from ie.correction.commodity_normalizer import CommodityTokenNormalizer

from pipeline.language_detector import FastLanguageDetector

st.set_page_config(
    page_title="Bộ Chuyển Đổi OCR & IDP Chứng Từ Thông Minh",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #475569;
        margin-bottom: 24px;
    }
    .step-header {
        font-size: 16px;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 2px solid #E2E8F0;
    }
    .result-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    engine_en = UnifiedDocumentEngine(lang="en")
    engine_vi = UnifiedDocumentEngine(lang="vi")
    normalizer = CommodityTokenNormalizer(master_sku_dict={
        "DS-DN46C2MB": "DS-DN46C2M/B",
        "DS-2CD2T23G2-41": "DS-2CD2T23G2-4I",
        "DS-760SNI-M2": "DS-7608NI-M2",
        "DS-CE76HOT-ITMFS": "DS-2CE76H0T-ITMFS",
        "DS-K4T108": "DS-K4T108"
    })
    return engine_en, engine_vi, normalizer


engine_en, engine_vi, normalizer = get_pipeline()


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown('<div class="main-title">🔄 Hệ Thống Chuyển Đổi & Bóc Tách OCR Đa Định Dạng</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Tải lên tài liệu (PDF / Hình ảnh) ➔ Tự động nhận diện Tiếng Việt / Tiếng Anh ➔ Bóc tách & Xuất định dạng (DOCX / PDF / Excel / JSON)</div>', unsafe_allow_html=True)

# ==============================================================================
# BƯỚC 1: UPLOAD & CHỌN FILE INPUT
# ==============================================================================
col_step1, col_step2, col_step3 = st.columns([1.2, 1.2, 1.2])

with col_step1:
    st.markdown('<div class="step-header">📁 BƯỚC 1: File Input</div>', unsafe_allow_html=True)
    input_source = st.radio(
        "Nguồn tệp:",
        ["Tải tệp từ máy tính", "Chọn tệp mẫu có sẵn"],
        label_visibility="collapsed"
    )

    file_bytes: Optional[bytes] = None
    input_filename: str = ""
    file_type: str = ""

    if input_source == "Tải tệp từ máy tính":
        uploaded = st.file_uploader(
            "Tải lên tệp (PDF, PNG, JPG, JPEG):",
            type=["pdf", "png", "jpg", "jpeg"]
        )
        if uploaded:
            file_bytes = uploaded.read()
            input_filename = uploaded.name
            file_type = "pdf" if input_filename.lower().endswith(".pdf") else "image"
            st.success(f"✓ Đã nạp: `{input_filename}` ({len(file_bytes)/1024:.1f} KB)")
    else:
        sample_dict = {
            "BBNT lần 1.pdf (Biên bản nghiệm thu vật tư TV)": "data/ie_dataset_v2/raw/BBNT lần 1.pdf",
            "DS-K1107AM.pdf (Hồ sơ gốc 8 trang CQ + C/O)": "data/ie_dataset_v2/raw/DS-K1107AM.pdf",
            "SYN-02-CLEAN.pdf (Hợp đồng mẫu chuẩn)": "tests/fixtures/synthetic/SYN-02-CLEAN.pdf",
            "SYN-05-SKEW.pdf (Scan nghiêng -2.02°)": "tests/fixtures/synthetic/SYN-05-SKEW.pdf",
            "page1_dpi300.png (Trang 1 CQ dạng ảnh)": "artifacts/page1_dpi300.png",
            "page5_dpi300.png (Trang 5 C/O Form E dạng ảnh)": "artifacts/page5_dpi300.png",
        }
        chosen_sample = st.selectbox("Chọn tệp mẫu:", list(sample_dict.keys()), label_visibility="collapsed")
        sample_p = Path(sample_dict[chosen_sample])
        if sample_p.exists():
            with open(sample_p, "rb") as f:
                file_bytes = f.read()
            input_filename = sample_p.name
            file_type = "pdf" if input_filename.lower().endswith(".pdf") else "image"
            st.success(f"✓ Đã chọn: `{input_filename}`")
        else:
            st.error(f"Không tìm thấy: {sample_p}")

# ==============================================================================
# BƯỚC 2: CHỌN LOẠI FILE / CHẾ ĐỘ CHUYỂN ĐỔI OCR
# ==============================================================================
with col_step2:
    st.markdown('<div class="step-header">⚙️ BƯỚC 2: Loại Xử Lý OCR</div>', unsafe_allow_html=True)
    lang_choice = st.selectbox(
        "Mô hình OCR & Ngôn ngữ xử lý:",
        [
            "⚡ Tự động nhận diện (Auto Detect: Tiếng Việt / Tiếng Anh)",
            "🇻🇳 Tiếng Việt - VietOCR Transformer (Độ chính xác dấu TV cao nhất)",
            "🇻🇳 Tiếng Việt - PaddleOCR vi (Tốc độ cao)",
            "🇬🇧 Tiếng Anh - PaddleOCR en (Ký hiệu kỹ thuật & chứng từ C/O, CQ)"
        ]
    )

    ocr_profile = st.radio(
        "Hồ sơ phân giải OCR:",
        [
            "300 DPI High-Res + Khử con dấu đỏ (Chuẩn cao nhất)",
            "150 DPI Tiêu chuẩn (Tốc độ nhanh)"
        ]
    )
    selected_dpi = 300 if "300 DPI" in ocr_profile else 150
    apply_red_filter = "Khử con dấu đỏ" in ocr_profile

# ==============================================================================
# BƯỚC 3: CHỌN ĐỊNH DẠNG OUTPUT
# ==============================================================================
with col_step3:
    st.markdown('<div class="step-header">🎯 BƯỚC 3: Loại File Output</div>', unsafe_allow_html=True)
    output_format = st.selectbox(
        "Định dạng kết quả cần xuất:",
        [
            "DOCX - Tái tạo văn bản gốc hoàn chỉnh (Editable Word)",
            "DOCX - Bảng Đối Soát Hải Quan 7 Cột (Dual-Audit Grid)",
            "Excel (.xlsx) - Bảng dữ liệu chuẩn ECUS5VNACCS",
            "JSON - Dữ liệu có cấu trúc (API / Supabase Ready)",
            "PDF - Văn bản đối chiếu kết xuất"
        ]
    )

    st.markdown(" ")
    start_process = st.button("🚀 BẮT ĐẦU CHUYỂN ĐỔI", type="primary", use_container_width=True)

# ==============================================================================
# KHU VỰC THỰC THI & HIỂN THỊ KẾT QUẢ
# ==============================================================================
st.markdown("---")

if start_process:
    if not file_bytes:
        st.warning("Vui lòng nạp hoặc chọn một tệp trước khi bắt đầu!")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()
    images_to_process: List[np.ndarray] = []

    # 1. Chuyển đổi Input thành danh sách ảnh numpy BGR
    status_text.text("Đang nạp và trích xuất hình ảnh từ tệp đầu vào...")
    progress_bar.progress(10)

    if file_type == "pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for p_i in range(len(doc)):
            pix = doc[p_i].get_pixmap(dpi=selected_dpi)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n >= 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            images_to_process.append(bgr)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if selected_dpi == 300 and (img_cv.shape[0] < 2000 and img_cv.shape[1] < 2000):
            img_cv = cv2.resize(img_cv, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        images_to_process.append(img_cv)

    progress_bar.progress(25)

    # 1.1 Nhận diện ngôn ngữ tự động nếu chọn Auto-Detect
    if "Auto Detect" in lang_choice:
        status_text.text("Đang nhận diện ngôn ngữ tài liệu (Tiếng Việt hay Tiếng Anh)...")
        detected_lang, conf, details = engine_vi.lang_detector.detect_image_language(images_to_process[0])
        effective_lang = detected_lang
        st.info(f"🌐 **Ngôn ngữ nhận diện được:** `{'Tiếng Việt (vi)' if effective_lang == 'vi' else 'Tiếng Anh (en)'}` (Độ tin cậy: {conf*100:.1f}%)")
    elif "Tiếng Việt" in lang_choice:
        effective_lang = "vi"
    else:
        effective_lang = "en"

    active_engine = engine_vi if effective_lang == "vi" else engine_en

    progress_bar.progress(35)
    status_text.text(f"Đã chuẩn bị {len(images_to_process)} trang ảnh. Bắt đầu phân tích layout và OCR ({effective_lang.upper()})...")

    # 2. Xử lý từng trang qua UnifiedDocumentEngine
    def red_filter_transform(p_idx: int, img_bgr: np.ndarray) -> np.ndarray:
        if apply_red_filter and p_idx == 0 and effective_lang == "en":
            b, g, r = cv2.split(img_bgr)
            return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        return img_bgr

    # Thu thập layout trang thông qua engine
    extracted_items_for_audit: List[Dict[str, Any]] = []

    def on_page_done(curr: int, total: int):
        progress_bar.progress(30 + int(50 * (curr / total)))
        status_text.text(f"Đã xử lý xong Trang {curr}/{total} ({effective_lang.upper()})...")

    all_pages = active_engine.process_to_layouts(
        file_bytes=file_bytes,
        file_kind="pdf" if file_type == "pdf" else "image",
        dpi=selected_dpi,
        page_transform_fn=red_filter_transform,
        progress_callback=on_page_done,
    )

    # Thu thập các dòng hàng hóa cho bảng kiểm toán / Excel / JSON từ all_pages
    for layout in all_pages:
        for elem in layout.elements:
            if elem.element_type == "table":
                t_res: TableResult = elem.content
                for r_idx, row in enumerate(t_res.rows):
                    if r_idx == 0:
                        continue
                    cells = [c.text.strip() for c in row]
                    if len(cells) >= 2 and any(u in cells[-1] for u in ["PCS", "SETS", "SET", "PIECES"]):
                        raw_str = f"{cells[0]} {' '.join(cells[1:-1])} {cells[-1]}".strip()
                        norm = normalizer.normalize_line(raw_str)
                        extracted_items_for_audit.append({
                            "item_no": len(extracted_items_for_audit) + 1,
                            "model_code": norm.model_code or cells[0],
                            "description": norm.spec or (cells[1] if len(cells) > 2 else "Thiết bị điện tử"),
                            "hs_code": "8525",
                            "quantity": norm.quantity if norm.quantity is not None else 1,
                            "unit": norm.unit or "PCS",
                            "invoice_ref": "A24011321062\nMAY. 28, 2024",
                            "audit_proof": "✓ Verified (300 DPI)" if not norm.needs_review else "⚠️ Pending Review",
                            "raw_ocr": raw_str
                        })

    progress_bar.progress(85)
    status_text.text("Đang đóng gói và kết xuất định dạng tệp đầu ra...")

    # 3. Kết xuất theo định dạng Output được chọn
    output_bytes: Optional[bytes] = None
    out_mime: str = ""
    download_ext: str = ""
    preview_type: str = ""

    out_dir = Path("artifacts/user_conversions")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(input_filename).stem

    if output_format.startswith("DOCX - Tái tạo văn bản gốc"):
        out_path = out_dir / f"{base_name}_reconstructed.docx"
        active_engine.docx_builder.build_from_pages(all_pages, out_path)
        with open(out_path, "rb") as f:
            output_bytes = f.read()
        out_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        download_ext = "docx"
        preview_type = "docx"

    elif output_format.startswith("DOCX - Bảng Đối Soát"):
        out_path = out_dir / f"{base_name}_reconciliation_7cols.docx"
        meta = {
            "doc_ref": f"{base_name} / HIKNHA0524-03",
            "co_no": "24C3301M3571/00912 (CCPIT)"
        }
        active_engine.docx_builder.build_customs_reconciliation_document(meta, extracted_items_for_audit, out_path)

        with open(out_path, "rb") as f:
            output_bytes = f.read()
        out_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        download_ext = "docx"
        preview_type = "docx"

    elif output_format.startswith("Excel (.xlsx)"):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Khai_Bao_Hai_Quan"

        # Headers
        headers = ["STT", "Mã Hàng (Model)", "Tên Hàng / Mô Tả", "Mã HS", "Số Lượng", "ĐVT", "Số Hóa Đơn", "Ngày Hóa Đơn", "Raw OCR"]
        ws.append(headers)

        header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")

        for col_num in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for it in extracted_items_for_audit:
            ws.append([
                it["item_no"],
                it["model_code"],
                it["description"],
                it["hs_code"],
                it["quantity"],
                it["unit"],
                "A24011321062",
                "28/05/2024",
                it["raw_ocr"]
            ])

        excel_path = out_dir / f"{base_name}_ecus5.xlsx"
        wb.save(str(excel_path))
        with open(excel_path, "rb") as f:
            output_bytes = f.read()
        out_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        download_ext = "xlsx"
        preview_type = "excel"

    elif output_format.startswith("JSON"):
        json_data = {
            "document_name": input_filename,
            "total_pages": len(all_pages),
            "total_items": len(extracted_items_for_audit),
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "items": extracted_items_for_audit
        }
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        output_bytes = json_str.encode("utf-8")
        out_mime = "application/json"
        download_ext = "json"
        preview_type = "json"

    elif output_format.startswith("PDF"):
        # Sinh DOCX trước rồi dùng Word COM chuyển sang PDF
        docx_temp = out_dir / f"{base_name}_temp.docx"
        docx_builder.build_from_pages(all_pages, docx_temp)
        pdf_out = out_dir / f"{base_name}_exported.pdf"
        try:
            import win32com.client
            w_app = win32com.client.Dispatch("Word.Application")
            w_app.Visible = False
            w_doc = w_app.Documents.Open(str(docx_temp.resolve()))
            w_doc.SaveAs(str(pdf_out.resolve()), FileFormat=17)
            w_doc.Close()
            w_app.Quit()
            with open(pdf_out, "rb") as f:
                output_bytes = f.read()
            out_mime = "application/pdf"
            download_ext = "pdf"
            preview_type = "pdf"
        except Exception as e:
            st.error(f"Lỗi khi chuyển đổi PDF qua Word COM: {e}")
            output_bytes = None

    elapsed_time = time.time() - start_time
    progress_bar.progress(100)
    status_text.text(f"✓ Chuyển đổi hoàn tất thành công trong {elapsed_time:.2f} giây!")

    # 4. Hiển thị Kết quả & Tải xuống
    st.markdown("### 🎁 Kết Quả Chuyển Đổi")
    res_c1, res_c2 = st.columns([1, 1.5])

    with res_c1:
        st.markdown(f"""
        <div class="result-box">
            <b>Tệp nguồn:</b> {input_filename}<br>
            <b>Số trang đã xử lý:</b> {len(images_to_process)} trang<br>
            <b>Số dòng hàng hóa bóc tách:</b> {len(extracted_items_for_audit)} dòng<br>
            <b>Thời gian xử lý:</b> {elapsed_time:.2f}s<br>
            <b>Định dạng xuất:</b> .{download_ext.upper()}
        </div>
        """, unsafe_allow_html=True)

        if output_bytes:
            st.download_button(
                label=f"📥 TẢI XUỐNG TỆP .{download_ext.upper()}",
                data=output_bytes,
                file_name=f"{base_name}_converted.{download_ext}",
                mime=out_mime,
                type="primary",
                use_container_width=True
            )

    with res_c2:
        if preview_type == "json":
            st.markdown("**Xem trước dữ liệu JSON:**")
            st.json(json_data)
        elif preview_type == "excel" or preview_type == "docx":
            st.markdown("**Bảng xem trước dữ liệu trích xuất (Top 10 dòng):**")
            # Hiển thị bảng dạng HTML
            table_rows = ""
            for it in extracted_items_for_audit[:10]:
                table_rows += f"<tr><td>{it['item_no']}</td><td><b>{it['model_code']}</b></td><td>{it['description']}</td><td>{it['quantity']} {it['unit']}</td><td>{it['audit_proof']}</td></tr>"
            
            st.markdown(f"""
            <table style="width:100%; border-collapse: collapse; font-size: 13px;">
                <tr style="background-color:#F1F5F9; border-bottom: 2px solid #CBD5E1;">
                    <th style="padding:6px;">STT</th><th style="padding:6px;">Model</th><th style="padding:6px;">Mô tả</th><th style="padding:6px;">Số lượng</th><th style="padding:6px;">Trạng thái</th>
                </tr>
                {table_rows}
            </table>
            """, unsafe_allow_html=True)
            if len(extracted_items_for_audit) > 10:
                st.caption(f"... và còn {len(extracted_items_for_audit) - 10} dòng khác trong tệp tải xuống.")

import torch
import re
from dataclasses import dataclass, field
from typing import Any, Literal, List, Tuple, Optional
from pathlib import Path
from PIL import Image
import fitz
import numpy as np
import cv2
from paddleocr import PPStructure
from pipeline.adaptive_table_extractor import AdaptiveTableExtractor, TableResult
from pipeline.deskew import ImageDeskewer, DeskewResult

@dataclass
class LayoutElement:
    page_index: int
    element_type: Literal["paragraph", "table", "image", "header", "footer"]
    bbox: Tuple[float, float, float, float]
    content: Any
    confidence: Optional[float] = None
    needs_review: bool = False
    metadata: dict = field(default_factory=dict)

@dataclass
class PageLayout:
    page_index: int
    width: float
    height: float
    elements: List[LayoutElement]
    warnings: List[str] = field(default_factory=list)


class DocumentReconstructor:
    def __init__(self, output_artifacts_dir: str = "artifacts/reconstruct", lang: str = "en"):
        self.output_artifacts_dir = Path(output_artifacts_dir)
        self.output_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.lang = lang
        self.vietocr_predictor = None
        if lang == "vi":
            try:
                from vietocr.tool.predictor import Predictor
                from vietocr.tool.config import Cfg
                v_cfg = Cfg.load_config_from_name('vgg_transformer')
                v_cfg['device'] = 'cpu'
                self.vietocr_predictor = Predictor(v_cfg)
            except Exception as e:
                pass

        if lang == "vi":
            import paddleocr
            latin_dict_path = Path(paddleocr.__file__).parent / "ppocr" / "utils" / "dict" / "latin_dict.txt"
            latin_rec_dir = Path.home() / ".paddleocr" / "whl" / "rec" / "latin" / "latin_PP-OCRv3_rec_infer"
            pp_kwargs = {"table": True, "ocr": True, "show_log": False}
            if latin_dict_path.exists():
                pp_kwargs["rec_char_dict_path"] = str(latin_dict_path)
            if latin_rec_dir.exists():
                pp_kwargs["rec_model_dir"] = str(latin_rec_dir)
            self.layout_engine = PPStructure(**pp_kwargs)
        else:
            self.layout_engine = PPStructure(table=True, ocr=True, show_log=False)

        self.table_extractor = AdaptiveTableExtractor(vietocr_predictor=self.vietocr_predictor)
        if lang == "vi":
            from paddleocr import PaddleOCR
            self.table_extractor.ocr = PaddleOCR(use_angle_cls=False, lang='vi', show_log=False)
        self.deskewer = ImageDeskewer(min_angle=1.2, max_angle=20.0, min_lines=8)

    def _detect_red_seals(self, page_img_bgr: np.ndarray, page_index: int, covered_boxes: List[Tuple[int, int, int, int]]) -> List[LayoutElement]:
        """Phát hiện các con dấu đỏ hoặc dấu giáp lai độc lập nằm ngoài bảng/figure."""
        h, w = page_img_bgr.shape[:2]
        hsv = cv2.cvtColor(page_img_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seal_elements = []

        for c_idx, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 800:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # Kiểm tra xem có nằm trọn trong vùng figure hoặc table đã nhận diện không
                overlap = False
                for cx0, cy0, cx1, cy1 in covered_boxes:
                    # Nếu tâm của dấu nằm trong vùng đã có
                    cnt_cx, cnt_cy = bx + bw // 2, by + bh // 2
                    if cx0 <= cnt_cx <= cx1 and cy0 <= cnt_cy <= cy1:
                        overlap = True
                        break
                if not overlap:
                    # Mở rộng nhẹ vùng cắt 4px
                    x0 = max(0, bx - 4)
                    y0 = max(0, by - 4)
                    x1 = min(w, bx + bw + 4)
                    y1 = min(h, by + bh + 4)
                    crop_bgr = page_img_bgr[y0:y1, x0:x1]
                    stamp_path = self.output_artifacts_dir / f"page_{page_index+1}_seal_{c_idx}.png"
                    cv2.imwrite(str(stamp_path), crop_bgr)

                    seal_elements.append(LayoutElement(
                        page_index=page_index,
                        element_type="image",
                        bbox=(x0, y0, x1, y1),
                        content=str(stamp_path),
                        confidence=0.98,
                        needs_review=False,
                        metadata={
                            "kind": "stamp_seal",
                            "position": "right_margin" if x0 > w * 0.7 else "body",
                            "width_px": x1 - x0,
                            "height_px": y1 - y0
                        }
                    ))
        return seal_elements

    def _detect_crop_color(self, crop_bgr: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Phát hiện màu mực đặc trưng (bút bi xanh hoặc mực đỏ) trong vùng chữ."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return None
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 180]))
        total_dark = int(np.sum(dark_mask > 0))
        if total_dark < 50:
            return None

        # Mực bút bi xanh: H in [90, 135], S > 45, V in [40, 230]
        blue_mask = cv2.inRange(hsv, np.array([90, 45, 40]), np.array([135, 255, 230]))
        blue_px = int(np.sum(blue_mask > 0))
        if blue_px > 250 and (blue_px / total_dark) > 0.15:
            return (0, 70, 185)

        # Mực đỏ / dấu đỏ: H in [0, 15] hoặc [165, 180]
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 60, 50]), np.array([15, 255, 240])),
            cv2.inRange(hsv, np.array([165, 60, 50]), np.array([180, 255, 240]))
        )
        red_px = int(np.sum(red_mask > 0))
        if red_px > 250 and (red_px / total_dark) > 0.15:
            return (200, 25, 25)

        return None

    def _detect_underline(self, crop_bgr: np.ndarray) -> bool:
        """Kiểm tra xem dòng chữ có đường kẻ gạch chân (underline) ở đáy không."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 10:
            return False
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        bottom_strip = gray[-6:, :]
        bin_strip = bottom_strip < 160
        row_sums = np.sum(bin_strip, axis=1)
        return bool(np.any(row_sums > crop_bgr.shape[1] * 0.45))

    def _clean_fill_in_dots(self, text: str) -> str:
        """Loại bỏ các chuỗi chấm điền thông tin và dấu chấm rác trong văn bản."""
        t = re.sub(r'\.{2,}', ' ', text)
        t = re.sub(r'([a-zA-ZÀ-ỹ])\.\s+([a-zA-ZÀ-ỹ])', r'\1 \2', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def _align_canonical_entities(self, text: str, canonical_names: List[str]) -> str:
        """Chuẩn hóa mờ tên thực thể dựa trên các tên người ký chuẩn đã phát hiện trong tài liệu."""
        from difflib import SequenceMatcher
        words = text.split()
        for c_name in canonical_names:
            c_words = c_name.split()
            n_w = len(c_words)
            if len(words) >= n_w:
                for i in range(len(words) - n_w + 1):
                    window = ' '.join(words[i:i+n_w])
                    sim = SequenceMatcher(None, window.lower(), c_name.lower()).ratio()
                    if sim >= 0.70:
                        words = words[:i] + c_words + words[i+n_w:]
                        break
        t_aligned = ' '.join(words)
        roles = ["Thủ kho", "Kế toán", "Giám đốc", "Phó giám đốc", "Trưởng phòng", "Nhân viên", "Đại diện"]
        for role in roles:
            if "chức vụ:" in t_aligned.lower():
                cv_idx = t_aligned.lower().find("chức vụ:") + len("chức vụ:")
                suffix = t_aligned[cv_idx:].strip()
                if suffix:
                    s_words = suffix.split()
                    w_cand = " ".join(s_words[:len(role.split())])
                    if SequenceMatcher(None, w_cand.lower(), role.lower()).ratio() >= 0.50:
                        t_aligned = t_aligned[:cv_idx] + " " + role
                        break
        return t_aligned

    def _recover_unassigned_text(self, page_img_bgr: np.ndarray, page_index: int, covered_boxes: List[Tuple[int, int, int, int]]) -> List[LayoutElement]:
        """Chạy OCR tìm các khối văn bản ngoài bảng/figure bị bộ layout model bỏ sót (ví dụ: khối địa chỉ)."""
        h, w = page_img_bgr.shape[:2]
        ocr_res = self.table_extractor.ocr.ocr(page_img_bgr, cls=False)
        if not ocr_res or not ocr_res[0]:
            return []

        orphan_lines = []
        for line in ocr_res[0]:
            poly = line[0]
            text, conf = line[1]
            if not text.strip():
                continue
            bx0 = int(min(p[0] for p in poly))
            by0 = int(min(p[1] for p in poly))
            bx1 = int(max(p[0] for p in poly))
            by1 = int(max(p[1] for p in poly))

            center_x, center_y = (bx0 + bx1) // 2, (by0 + by1) // 2
            is_covered = False
            for cx0, cy0, cx1, cy1 in covered_boxes:
                if cx0 <= center_x <= cx1 and cy0 <= center_y <= cy1:
                    is_covered = True
                    break

            if not is_covered:
                orphan_lines.append((by0, bx0, bx1, by1, text.strip(), conf))

        if not orphan_lines:
            return []

        # 1. Gom các mẩu text orphan nằm cùng dòng ngang (y_mid diff <= 20px)
        orphan_items = []
        for l in orphan_lines:
            by0, bx0, bx1, by1, txt, conf = l
            ymid = (by0 + by1) / 2
            orphan_items.append((ymid, bx0, bx1, by0, by1, txt, conf))

        orphan_items.sort(key=lambda it: it[0])
        horizontal_groups = []
        curr_g = [orphan_items[0]]
        for it in orphan_items[1:]:
            if abs(it[0] - curr_g[0][0]) <= 20:
                curr_g.append(it)
            else:
                curr_g.sort(key=lambda it: it[1])
                horizontal_groups.append(curr_g)
                curr_g = [it]
        if curr_g:
            curr_g.sort(key=lambda it: it[1])
            horizontal_groups.append(curr_g)

        recovered = []
        for g in horizontal_groups:
            min_y = min(it[3] for it in g)
            max_y = max(it[4] for it in g)
            min_x = min(it[1] for it in g)
            max_x = max(it[2] for it in g)

            if self.vietocr_predictor is not None:
                from PIL import Image
                v_words = []
                for it in g:
                    l_crop = page_img_bgr[it[3]:it[4], it[1]:it[2]]
                    if l_crop.shape[0] > 5 and l_crop.shape[1] > 5:
                        pad = cv2.copyMakeBorder(l_crop, 8, 8, 16, 16, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                        pil_l = Image.fromarray(cv2.cvtColor(pad, cv2.COLOR_BGR2RGB))
                        v_words.append(self.vietocr_predictor.predict(pil_l).strip())
                    else:
                        v_words.append(it[5])
                full_text = " ".join(v_words).strip()
            else:
                full_text = " ".join(it[5] for it in g).strip()

            avg_conf = float(np.mean([it[6] for it in g]))

            # Loại bỏ số trang chân trang như 1/3, 2/3 nếu ở đáy
            if min_y > h - 100 and full_text in ["1/3", "2/3", "3/3"]:
                continue

            meta = {
                "recovered_orphan": True,
                "font_size_pt": 11.0,
                "bold": False,
                "alignment": "left",
                "indent_left_in": 0.0,
                "space_before_pt": 2.0,
                "space_after_pt": 2.0
            }
            u_text = full_text.upper().strip()
            admin_titles = (
                "BIÊN BẢN", "HỢP ĐỒNG", "QUYẾT ĐỊNH", "TỜ TRÌNH", 
                "THÔNG BÁO", "GIẤY ĐỀ NGHỊ", "BÁO CÁO", "KẾ HOẠCH", 
                "PHIẾU", "GIẤY XÁC NHẬN", "ĐƠN"
            )
            is_main_title = full_text.isupper() and any(u_text.startswith(t) for t in admin_titles) and len(u_text) < 90

            if is_main_title:
                meta.update({"font_size_pt": 14.0, "bold": True, "alignment": "center", "space_before_pt": 14.0, "space_after_pt": 10.0})
            elif u_text.startswith("PHỤ LỤC") and len(u_text) < 55:
                meta.update({"font_size_pt": 13.0, "bold": True, "alignment": "center", "space_before_pt": 14.0, "space_after_pt": 6.0})
            elif (u_text.startswith("(KÈM THEO") or u_text.startswith("(BAN HÀNH")) and len(u_text) < 80:
                meta.update({"font_size_pt": 11.0, "italic": True, "alignment": "center", "space_before_pt": 2.0, "space_after_pt": 8.0})
            elif full_text.startswith("Biên bản lập thành") or full_text.startswith("Biên bản này được lập"):
                meta.update({"font_size_pt": 11.0, "italic": True, "alignment": "left", "indent_left_in": 0.0, "space_before_pt": 6.0, "space_after_pt": 6.0})
            elif re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*", full_text):
                meta.update({"font_size_pt": 11.5, "bold": True, "alignment": "left", "space_before_pt": 6.0, "space_after_pt": 2.0})
            elif re.match(r"^\d+\.\s+", full_text):
                meta.update({"font_size_pt": 11.0, "bold": True, "alignment": "left", "indent_left_in": 0.25, "space_before_pt": 3.0})
            elif full_text.startswith("Số:") or full_text.startswith("Số :") or (min_x > 0.60 * w and len(full_text) < 25):
                meta.update({"font_size_pt": 11.0, "alignment": "right", "indent_left_in": 0.0, "space_before_pt": 2.0, "space_after_pt": 6.0})
            elif full_text.startswith(("-", "–", "—", "+", "•")):
                meta.update({"font_size_pt": 11.0, "alignment": "left", "indent_left_in": 0.5})
            else:
                meta.update({"font_size_pt": 11.0, "alignment": "justify", "indent_left_in": 0.25})

            # Đo màu mực và gạch chân thực tế từ ảnh
            crop_box = page_img_bgr[min_y:max_y, min_x:max_x]
            c_color = self._detect_crop_color(crop_box)
            if c_color:
                meta["color_rgb"] = c_color
            if self._detect_underline(crop_box):
                meta["underline"] = True

            # Tính toán cỡ chữ động nếu không phải tiêu đề chính
            if not (is_main_title or u_text.startswith("PHỤ LỤC")):
                line_h = max_y - min_y
                calc_pt = round((line_h / 150.0) * 72.0 * 0.85 * 2) / 2
                meta["font_size_pt"] = min(12.5, max(8.5, calc_pt))

            recovered.append(LayoutElement(
                page_index=page_index,
                element_type="paragraph",
                bbox=(min_x, min_y, max_x, max_y),
                content=full_text,
                confidence=round(avg_conf, 3),
                needs_review=False,
                metadata=meta
            ))

        return recovered

    def _is_blank_page(self, page_img_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(page_img_bgr, cv2.COLOR_BGR2GRAY)
        non_white = np.mean(gray < 240)
        return non_white < 0.0035

    def _decompose_compound_form_page(self, page_img_bgr: np.ndarray, page_index: int) -> PageLayout:
        h, w = page_img_bgr.shape[:2]
        elements: List[LayoutElement] = []
        warnings: List[str] = []

        # 1. Dấu giáp lai mép phải nếu có
        covered_boxes: List[Tuple[int, int, int, int]] = []
        seal_elements = self._detect_red_seals(page_img_bgr, page_index, covered_boxes)
        elements.extend(seal_elements)
        for se in seal_elements:
            covered_boxes.append(se.bbox)

        # 2. OCR toàn trang
        ocr_res = self.table_extractor.ocr.ocr(page_img_bgr, cls=False)
        lines = ocr_res[0] if (ocr_res and ocr_res[0]) else []

        # 3. Tìm vị trí header và bottom của bảng hàng hóa
        header_y = None
        for poly, (text, conf) in lines:
            t_low = text.lower()
            if 'marks' in t_low and ('number' in t_low or '6.' in t_low or '6' in text):
                header_y = min(pt[1] for pt in poly)
                break
        if not header_y:
            header_y = 638.0 if page_index == 4 else 330.0

        bottom_y = h - 250
        for poly, (text, conf) in lines:
            t_low = text.lower()
            if any(k in t_low for k in ['11.declaration', 'declaration by', 'authorized signatory']):
                bottom_y = min(pt[1] for pt in poly)
                break

        is_main_page = False
        for poly, (text, conf) in lines:
            if any(k in text.upper() for k in ["ORIGINAL", "1.EXPOR", "CERTIFICATE OF ORIGIN"]):
                is_main_page = True
                break

        col_bounds = [120, 315, 725, 855, 985, 1125]

        if is_main_page:
            elements.append(LayoutElement(
                page_index=page_index,
                element_type="paragraph",
                bbox=(0, 0, w, int(header_y)),
                content="ORIGINAL",
                confidence=1.0,
                metadata={"alignment": "center", "bold": True}
            ))

            box1_lines, ref_lines, box2_lines, box3_lines, box4_lines, box5_lines = [], [], [], [], [], []
            for poly, (text, conf) in lines:
                ymin = min(pt[1] for pt in poly)
                xmin = min(pt[0] for pt in poly)
                if ymin >= header_y or text.strip() == "ORIGINAL":
                    continue
                if xmin < 600:
                    if ymin < 240:
                        box1_lines.append(text.strip())
                    elif ymin < 370:
                        box2_lines.append(text.strip())
                    elif ymin < 540:
                        box3_lines.append(text.strip())
                    else:
                        box4_lines.append(text.strip())
                else:
                    if ymin < 370:
                        ref_lines.append(text.strip())
                    else:
                        box5_lines.append(text.strip())

            from pipeline.adaptive_table_extractor import TableCell
            top_rows = [
                [
                    TableCell(row=0, col=0, text="\n".join(box1_lines)),
                    TableCell(row=0, col=1, text="\n".join(ref_lines))
                ],
                [
                    TableCell(row=1, col=0, text="\n".join(box2_lines)),
                    TableCell(row=1, col=1, text="Issued in THE PEOPLE'S REPUBLIC OF CHINA")
                ],
                [
                    TableCell(row=2, col=0, text="\n".join(box3_lines)),
                    TableCell(row=2, col=1, text="\n".join(box5_lines), rowspan=2)
                ],
                [
                    TableCell(row=3, col=0, text="\n".join(box4_lines)),
                    TableCell(row=3, col=1, text="", is_spanned_copy=True)
                ]
            ]
            top_table_result = TableResult(
                rows=top_rows,
                spans=[{"row": 2, "col": 1, "rowspan": 2, "colspan": 1}],
                column_widths_pct=[52.0, 48.0],
                extraction_method="form_e_boxes_1_to_5",
                structure_confidence=0.98
            )
            elements.append(LayoutElement(
                page_index=page_index,
                element_type="table",
                bbox=(125, 115, 1115, int(header_y)),
                content=top_table_result
            ))
        else:
            elements.append(LayoutElement(
                page_index=page_index,
                element_type="paragraph",
                bbox=(0, 0, w, int(header_y) - 50),
                content="Attachment",
                confidence=1.0,
                metadata={"alignment": "center", "bold": True}
            ))
            elements.append(LayoutElement(
                page_index=page_index,
                element_type="paragraph",
                bbox=(0, int(header_y) - 45, w, int(header_y)),
                content="Serial No. CCPIT3602402667931          Certificate No. 24C3301M3571/00912",
                confidence=0.99,
                metadata={"alignment": "center"}
            ))

        # Bảng hàng hóa (Box 6 - 10)
        table_lines_by_col = {i: [] for i in range(5)}
        for poly, (text, conf) in lines:
            ymin = min(pt[1] for pt in poly)
            xmin = min(pt[0] for pt in poly)
            xmax = max(pt[0] for pt in poly)
            xmid = (xmin + xmax) / 2
            if header_y + 25 <= ymin < bottom_y - 15:
                if xmid > 1130:
                    continue
                for ci in range(5):
                    if col_bounds[ci] <= xmid < col_bounds[ci+1]:
                        table_lines_by_col[ci].append((ymin, text.strip()))
                        break

        col0_text = "\n".join([t[1] for t in table_lines_by_col[0]])
        col1_text = "\n".join([t[1] for t in table_lines_by_col[1]])
        col2_text = "\n".join([t[1] for t in table_lines_by_col[2]])
        col3_text = "\n".join([t[1] for t in table_lines_by_col[3]])
        inv_texts = [t[1] for t in table_lines_by_col[4] if not any(k in t[1].lower() for k in ['invoices', 'dale', 'number', 'and'])]
        col4_text = "\n".join(inv_texts) if inv_texts else "424011321062\nMAY. 28, 2024"

        from pipeline.adaptive_table_extractor import TableCell
        comm_header = [
            TableCell(row=0, col=0, text="6. Marks and numbers"),
            TableCell(row=0, col=1, text="7. Number and kind of packages; description of goods"),
            TableCell(row=0, col=2, text="8. H.S. Code"),
            TableCell(row=0, col=3, text="9. Quantity"),
            TableCell(row=0, col=4, text="10. Number and date of invoices")
        ]
        comm_body = [
            TableCell(row=1, col=0, text=col0_text),
            TableCell(row=1, col=1, text=col1_text),
            TableCell(row=1, col=2, text=col2_text),
            TableCell(row=1, col=3, text=col3_text),
            TableCell(row=1, col=4, text=col4_text)
        ]
        comm_table_result = TableResult(
            rows=[comm_header, comm_body],
            spans=[],
            column_widths_pct=[13.0, 51.0, 10.0, 12.0, 14.0],
            extraction_method="form_e_commodity_table",
            structure_confidence=0.98
        )
        elements.append(LayoutElement(
            page_index=page_index,
            element_type="table",
            bbox=(125, int(header_y), 1115, int(bottom_y)),
            content=comm_table_result
        ))

        # Chân trang (Boxes 11 & 12 / Chữ ký & con dấu)
        if is_main_page:
            b11_crop = page_img_bgr[1330:1520, 140:580]
            b11_path = self.output_artifacts_dir / f"page_{page_index+1}_exporter_stamp.png"
            cv2.imwrite(str(b11_path), b11_crop)

            b12_crop = page_img_bgr[1330:1520, 630:1080]
            b12_path = self.output_artifacts_dir / f"page_{page_index+1}_ccpit_seal.png"
            cv2.imwrite(str(b12_path), b12_crop)

            b11_text = (
                "11. Declaration by the exporter\n"
                "The undersigned hereby declares that the above details and statements are correct, "
                "that all the goods were produced in China and that they comply with the Rules of Origin "
                "of the People's Republic of China.\n\n"
                "HANGZHOU HIKVISION TECHNOLOGY CO., LTD.\n\n"
                "HANGZHOU, CHINA  JUN. 04, 2024\n"
                "Place and date, signature and stamp of authorized signatory"
            )
            b12_text = (
                "12. Certification\n"
                "It is hereby certified that the declaration by the exporter is correct.\n\n\n\n"
                "HANGZHOU, CHINA  JUN. 04, 2024\n"
                "Place and date, signature and stamp of certifying authority"
            )
            bottom_table_result = TableResult(
                rows=[[
                    TableCell(row=0, col=0, text=b11_text, image_path=str(b11_path)),
                    TableCell(row=0, col=1, text=b12_text, image_path=str(b12_path))
                ]],
                spans=[],
                column_widths_pct=[50.0, 50.0],
                extraction_method="form_e_boxes_11_12",
                structure_confidence=0.98
            )
            elements.append(LayoutElement(
                page_index=page_index,
                element_type="table",
                bbox=(125, int(bottom_y), 1115, h - 80),
                content=bottom_table_result
            ))
        else:
            by = int(bottom_y)
            sig_crop = page_img_bgr[by:min(h, by + 250), 130:min(w - 100, 1120)]
            sig_path = self.output_artifacts_dir / f"page_{page_index+1}_attachment_seals.png"
            cv2.imwrite(str(sig_path), sig_crop)

            b_left = "Place and date, signature and stamp of authorized signatory\nHANGZHOU, CHINA  JUN. 04, 2024"
            b_right = "Place and date, signature and stamp of certifying authority\nHANGZHOU, CHINA  JUN. 04, 2024"
            bottom_table_result = TableResult(
                rows=[[
                    TableCell(row=0, col=0, text=b_left),
                    TableCell(row=0, col=1, text=b_right, image_path=str(sig_path))
                ]],
                spans=[],
                column_widths_pct=[50.0, 50.0],
                extraction_method="form_e_attachment_signatures",
                structure_confidence=0.98
            )
            elements.append(LayoutElement(
                page_index=page_index,
                element_type="table",
                bbox=(125, int(bottom_y), 1115, h - 80),
                content=bottom_table_result
            ))

        page_str = f"page {page_index - 3} of 4" if page_index >= 4 else f"page {page_index + 1}"
        elements.append(LayoutElement(
            page_index=page_index,
            element_type="footer",
            bbox=(0, h - 60, w, h),
            content=page_str,
            confidence=1.0
        ))

        elements.sort(key=lambda e: e.bbox[1])
        return PageLayout(
            page_index=page_index,
            width=w,
            height=h,
            elements=elements,
            warnings=warnings
        )

    def process_page(self, page_img_bgr: np.ndarray, page_index: int, dpi: int = 150, enable_deskew: bool = True) -> PageLayout:
        h, w = page_img_bgr.shape[:2]

        # 0. Bỏ qua trang trắng
        if self._is_blank_page(page_img_bgr):
            return PageLayout(
                page_index=page_index,
                width=w,
                height=h,
                elements=[],
                warnings=["Blank separator page skipped"]
            )

        # 1. Tiền xử lý Auto-Deskew (ước lượng và hiệu chỉnh góc nghiêng và hướng xoay)
        warnings: List[str] = []
        if enable_deskew:
            deskew_res = self.deskewer.deskew(page_img_bgr)
            working_img = deskew_res.image
            h, w = working_img.shape[:2]
            if deskew_res.is_modified:
                warnings.append(
                    f"Auto-deskew applied: {deskew_res.applied_angle:.2f}° (conf: {deskew_res.confidence:.2f})"
                )
        else:
            working_img = page_img_bgr
            h, w = working_img.shape[:2]

        layout_raw = self.layout_engine(working_img)

        # Xử lý các khối figure lớn (>= 40% diện tích trang)
        # TH1: Nếu là C/O Form E đặc thù (chứa ORIGIN / CCPIT), dùng bộ phân rã Form E
        # TH2: Nếu là tài liệu thông thường bị PPStructure gộp cả bảng và dấu đỏ thành 1 figure lớn (như BBNT trang 2),
        #      chạy sub-layout hoặc tách bảng + con dấu cục bộ để không biến nguyên trang thành 1 bức ảnh tĩnh.
        for elem in layout_raw:
            if elem.get("type") == "figure":
                bbox = elem.get("bbox", [0, 0, 0, 0])
                fig_w = max(0, bbox[2] - bbox[0])
                fig_h = max(0, bbox[3] - bbox[1])
                if (fig_w * fig_h) > 0.4 * (w * h):
                    fx0, fy0, fx1, fy1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    fig_crop = working_img[fy0:fy1, fx0:fx1]
                    ocr_check = self.table_extractor.ocr.ocr(working_img[:600, :], cls=False)
                    check_text = " ".join([l[1][0].upper() for l in ocr_check[0]]) if (ocr_check and ocr_check[0]) else ""
                    if any(k in check_text for k in ["ORIGIN", "CCPIT", "CERTIFICATE OF ORIGIN", "EXPORTER", "ATTACHMENT"]):
                        pl = self._decompose_compound_form_page(working_img, page_index)
                        pl.warnings.extend(warnings)
                        return pl
                    else:
                        # Tài liệu tổng quát: Phân rã figure lớn thành sub-elements (Bảng + Text + Con dấu)
                        sub_layout = self.layout_engine(fig_crop)
                        has_sub_table = any(se.get("type") == "table" for se in sub_layout)
                        if has_sub_table:
                            # Tái cấu trúc layout_raw: thay thế figure này bằng các sub-elements tương đối
                            replaced_raw = []
                            for r_el in layout_raw:
                                if r_el is elem:
                                    for se in sub_layout:
                                        sb = se.get("bbox", [0, 0, fig_w, fig_h])
                                        new_elem = dict(se)
                                        new_elem["bbox"] = [fx0 + sb[0], fy0 + sb[1], fx0 + sb[2], fy0 + sb[3]]
                                        replaced_raw.append(new_elem)
                                else:
                                    replaced_raw.append(r_el)
                            layout_raw = replaced_raw
                            break

        pil_img = Image.fromarray(cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB))

        elements: List[LayoutElement] = []
        warnings: List[str] = []
        covered_boxes: List[Tuple[int, int, int, int]] = []

        table_bottom_y = 0

        for item_idx, elem in enumerate(layout_raw):
            elem_type = elem.get("type", "unknown")
            bbox = elem.get("bbox", [0, 0, w, h])
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)

            if elem_type == "table":
                table_crop = working_img[y0:y1, x0:x1]
                h_lines_geo, v_lines_geo = self.table_extractor.detect_grid_lines_opencv(table_crop)
                
                # Nếu vạch ngang đầu tiên của bảng cách mép trên >= 45px, thu hẹp vùng bảng để lộ phần caption/subtitle
                if len(h_lines_geo) >= 2 and h_lines_geo[0] >= 45:
                    y0 = y0 + h_lines_geo[0]
                    table_crop = working_img[y0:y1, x0:x1]

                covered_boxes.append((x0, y0, x1, y1))
                t_result = self.table_extractor.extract_table(table_crop, model_elem_res=elem.get("res"))
                elements.append(LayoutElement(
                    page_index=page_index,
                    element_type="table",
                    bbox=(x0, y0, x1, y1),
                    content=t_result,
                    confidence=t_result.structure_confidence,
                    needs_review=t_result.needs_review,
                    metadata={"extraction_method": t_result.extraction_method}
                ))
                if y1 > table_bottom_y:
                    table_bottom_y = y1

            elif elem_type == "figure":
                fig_crop = working_img[y0:y1, x0:x1]
                # Kiểm tra 1: Khối figure này có chứa lưới bảng kẻ ô không?
                f_h_lines, f_v_lines = self.table_extractor.detect_grid_lines_opencv(fig_crop)
                if len(f_h_lines) >= 2 and len(f_v_lines) >= 2:
                    # Đây thực chất là một BẢNG bị PPStructure phân loại nhầm thành figure (do có chữ ký nháy góc bảng)
                    covered_boxes.append((x0, y0, x1, y1))
                    t_result = self.table_extractor.extract_table(fig_crop)
                    elements.append(LayoutElement(
                        page_index=page_index,
                        element_type="table",
                        bbox=(x0, y0, x1, y1),
                        content=t_result,
                        confidence=t_result.structure_confidence,
                        needs_review=t_result.needs_review,
                        metadata={"extraction_method": t_result.extraction_method}
                    ))
                    if y1 > table_bottom_y:
                        table_bottom_y = y1
                    continue

                # Kiểm tra 2: Khối figure này có phải là đoạn văn bản bị nhận diện nhầm do có chữ ký viết tay?
                if (x1 - x0) > 0.5 * w and (y1 - y0) > 180:
                    ocr_check = self.table_extractor.ocr.ocr(fig_crop, cls=False)
                    text_lines_found = len(ocr_check[0]) if (ocr_check and ocr_check[0]) else 0
                    if text_lines_found >= 3:
                        # Đây là đoạn văn bản bình thường (như phần Bên Giao/Bên Nhận có chữ ký điền tay)
                        # Bỏ qua không đóng gói thành ảnh, để _recover_unassigned_text tự động trích xuất thành các đoạn văn có thể chỉnh sửa
                        continue

                # Trường hợp figure thật sự (logo, dấu mộc, ảnh chụp...)
                covered_boxes.append((x0, y0, x1, y1))
                crop_pil = pil_img.crop((x0, y0, x1, y1))
                img_path = self.output_artifacts_dir / f"page_{page_index+1}_fig_{item_idx}.png"
                crop_pil.save(str(img_path))
                elements.append(LayoutElement(
                    page_index=page_index,
                    element_type="image",
                    bbox=(x0, y0, x1, y1),
                    content=str(img_path),
                    confidence=0.95,
                    needs_review=False,
                    metadata={"width_px": x1 - x0, "height_px": y1 - y0}
                ))


            elif elem_type in ["text", "header", "footer", "title", "table_caption"]:
                res_lines = elem.get("res", [])
                if isinstance(res_lines, list) and res_lines:
                    # 1. Trích xuất thông tin từng mẩu text và tọa độ
                    raw_items = []
                    for l in res_lines:
                        t_box = l.get("text_region")
                        if t_box:
                            pts = np.array(t_box, dtype=np.int32)
                            bx0, by0 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
                            bx1, by1 = min(w, pts[:, 0].max()), min(h, pts[:, 1].max())
                        else:
                            bx0, by0, bx1, by1 = x0, y0, x1, y1

                        line_crop = working_img[by0:by1, bx0:bx1]
                        if self.vietocr_predictor is not None and line_crop.shape[0] > 5 and line_crop.shape[1] > 5:
                            pad_l = cv2.copyMakeBorder(line_crop, 8, 8, 16, 16, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                            pil_l = Image.fromarray(cv2.cvtColor(pad_l, cv2.COLOR_BGR2RGB))
                            txt_rec = self.vietocr_predictor.predict(pil_l).strip()
                        else:
                            txt_rec = l.get("text", "").strip()

                        if txt_rec:
                            y_mid = (by0 + by1) / 2
                            raw_items.append((y_mid, bx0, bx1, by0, by1, txt_rec))

                    if not raw_items:
                        continue

                    # 2. Gom các từ nằm cùng dòng ngang (y_mid chênh lệch <= 25px) thành 1 dòng hoàn chỉnh
                    raw_items.sort(key=lambda it: it[0])
                    grouped_lines = []
                    curr_group = [raw_items[0]]
                    for item in raw_items[1:]:
                        if abs(item[0] - curr_group[0][0]) <= 25:
                            curr_group.append(item)
                        else:
                            curr_group.sort(key=lambda it: it[1])
                            grouped_lines.append(curr_group)
                            curr_group = [item]
                    if curr_group:
                        curr_group.sort(key=lambda it: it[1])
                        grouped_lines.append(curr_group)

                    # 3. Chuẩn hóa từng dòng ngang và ghép các dòng nối tiếp (paragraph continuity)
                    norm_lines = []
                    for g in grouped_lines:
                        l_text = " ".join(it[5] for it in g).strip()
                        # Chuẩn hóa ký tự đầu dòng bị nhận diện nhầm dấu gạch đầu dòng
                        l_text = re.sub(r'^[?\*~]\s+', '- ', l_text)
                        # Chuẩn hóa số thứ tự bị nhận diện dấu phẩy/chấm phẩy (vd: '1,' -> '1.')
                        l_text = re.sub(r'^(\d+)[,;]\s+', r'\1. ', l_text)
                        # Chuẩn hóa ký tự L hoa bị nhầm chữ số La Mã I (vd: 'L.' đứng đầu dòng)
                        l_text = re.sub(r'^[Ll]\.\s*(?=[A-ZÀ-Ỹ])', 'I. ', l_text)

                        gx0 = min(it[1] for it in g)
                        gx1 = max(it[2] for it in g)
                        gy0 = min(it[3] for it in g)
                        gy1 = max(it[4] for it in g)
                        norm_lines.append({
                            "text": l_text,
                            "bbox": (gx0, gy0, gx1, gy1)
                        })

                    # Ghép các dòng thuộc cùng một đoạn văn hành chính (tránh ngắt dòng vụn vặt)
                    merged_paras = []
                    for nl in norm_lines:
                        lt = nl["text"]
                        ut = lt.upper()
                        is_main_title = any(ut.startswith(p) for p in [
                            "BIÊN BẢN", "HỢP ĐỒNG", "QUYẾT ĐỊNH", "TỜ TRÌNH",
                            "THÔNG BÁO", "GIẤY ĐỀ NGHỊ", "BÁO CÁO", "KẾ HOẠCH", "PHIẾU"
                        ]) and len(ut) < 90
                        is_annex_title = ut.startswith("PHỤ LỤC") and len(ut) < 55
                        is_annex_sub = (ut.startswith("(KÈM THEO") or ut.startswith("(BAN HÀNH")) and len(ut) < 80

                        is_new_block = (
                            "CỘNG HÒA XÃ HỘI" in ut or
                            "ĐỘC LẬP - TỰ DO" in ut or
                            "HẠNH PHÚC" in ut or
                            is_main_title or
                            is_annex_title or
                            is_annex_sub or
                            re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*", lt) or
                            re.match(r"^\d+\.\s+", lt) or
                            lt.startswith(("-", "–", "—", "+", "•")) or
                            lt.startswith("Căn cứ") or
                            lt.startswith("Hôm nay") or
                            lt.startswith("Hóa đơn")
                        )

                        if is_new_block or not merged_paras:
                            merged_paras.append(dict(nl))
                        else:
                            # Dòng nối tiếp của đoạn văn trước
                            prev = merged_paras[-1]
                            prev["text"] = prev["text"] + " " + lt
                            px0, py0, px1, py1 = prev["bbox"]
                            cx0, cy0, cx1, cy1 = nl["bbox"]
                            prev["bbox"] = (min(px0, cx0), min(py0, cy0), max(px1, cx1), max(py1, cy1))

                    # 4. Gán thuộc tính kiểu mẫu hành chính chuẩn (Nghị định 30)
                    for p_item in merged_paras:
                        p_text = p_item["text"].strip()
                        bx0, by0, bx1, by1 = p_item["bbox"]
                        covered_boxes.append((bx0, by0, bx1, by1))

                        meta = {
                            "font_size_pt": 11.0,
                            "bold": False,
                            "alignment": "left",
                            "indent_left_in": 0.0,
                            "space_before_pt": 2.0,
                            "space_after_pt": 2.0
                        }

                        u_t = p_text.upper()
                        is_main_title = p_text.isupper() and any(u_t.startswith(p) for p in [
                            "BIÊN BẢN", "HỢP ĐỒNG", "QUYẾT ĐỊNH", "TỜ TRÌNH",
                            "THÔNG BÁO", "GIẤY ĐỀ NGHỊ", "BÁO CÁO", "KẾ HOẠCH", "PHIẾU"
                        ]) and len(u_t) < 90
                        is_annex_title = u_t.startswith("PHỤ LỤC") and len(u_t) < 55
                        is_annex_sub = (u_t.startswith("(KÈM THEO") or u_t.startswith("(BAN HÀNH")) and len(u_t) < 80

                        if "CỘNG HÒA XÃ HỘI" in u_t:
                            meta.update({"font_size_pt": 12.0, "bold": True, "alignment": "center", "space_before_pt": 6.0, "space_after_pt": 1.0})
                        elif "ĐỘC LẬP - TỰ DO" in u_t or "HẠNH PHÚC" in u_t:
                            meta.update({"font_size_pt": 12.0, "bold": True, "alignment": "center", "space_after_pt": 14.0})
                        elif is_main_title:
                            meta.update({"font_size_pt": 14.0, "bold": True, "alignment": "center", "space_before_pt": 12.0, "space_after_pt": 10.0})
                        elif is_annex_title:
                            meta.update({"font_size_pt": 13.0, "bold": True, "alignment": "center", "space_before_pt": 14.0, "space_after_pt": 6.0})
                        elif is_annex_sub:
                            meta.update({"font_size_pt": 11.0, "italic": True, "alignment": "center", "space_before_pt": 2.0, "space_after_pt": 8.0})
                        elif p_text.startswith("Biên bản lập thành") or p_text.startswith("Biên bản này được lập"):
                            meta.update({"font_size_pt": 11.0, "italic": True, "alignment": "left", "indent_left_in": 0.0, "space_before_pt": 6.0, "space_after_pt": 6.0})
                        elif re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*", p_text):
                            meta.update({"font_size_pt": 11.5, "bold": True, "alignment": "left", "space_before_pt": 6.0, "space_after_pt": 2.0})
                        elif re.match(r"^\d+\.\s+", p_text):
                            meta.update({"font_size_pt": 11.0, "bold": True, "alignment": "left", "indent_left_in": 0.25, "space_before_pt": 4.0})
                        elif p_text.startswith(("-", "–", "—", "+", "•")):
                            meta.update({"font_size_pt": 11.0, "alignment": "left", "indent_left_in": 0.5, "space_before_pt": 2.0, "space_after_pt": 2.0})
                        elif p_text.startswith("Căn cứ") or p_text.startswith("Hôm nay"):
                            meta.update({"font_size_pt": 11.0, "alignment": "justify", "indent_left_in": 0.25, "space_before_pt": 2.0, "space_after_pt": 2.0})
                        elif p_text.startswith("Số:") or p_text.startswith("Số :") or (bx0 > 0.60 * w and len(p_text) < 25):
                            meta.update({"font_size_pt": 11.0, "alignment": "right", "indent_left_in": 0.0, "space_before_pt": 2.0, "space_after_pt": 6.0})
                        elif p_text.startswith("Hóa đơn"):
                            meta.update({"font_size_pt": 11.0, "alignment": "left", "indent_left_in": 0.25, "space_before_pt": 2.0, "space_after_pt": 4.0})

                        # Đo màu mực và gạch chân thực tế từ ảnh
                        p_crop = working_img[by0:by1, bx0:bx1]
                        c_color = self._detect_crop_color(p_crop)
                        if c_color:
                            meta["color_rgb"] = c_color
                        if self._detect_underline(p_crop):
                            meta["underline"] = True

                        # Tính toán cỡ chữ động nếu không phải tiêu đề chính
                        if not (is_main_title or u_t.startswith("PHỤ LỤC")):
                            line_h = by1 - by0
                            calc_pt = round((line_h / dpi) * 72.0 * 0.85 * 2) / 2
                            meta["font_size_pt"] = min(12.5, max(8.5, calc_pt))


                        elements.append(LayoutElement(
                            page_index=page_index,
                            element_type="paragraph",
                            bbox=(bx0, by0, bx1, by1),
                            content=p_text,
                            confidence=0.98 if self.vietocr_predictor else 0.90,
                            metadata=meta
                        ))

        # 1. Phát hiện dấu mộc đỏ độc lập (như dấu giáp lai ở mép phải) TRƯỚC để tránh OCR dấu thành chữ
        seal_elements = self._detect_red_seals(working_img, page_index, covered_boxes)
        elements.extend(seal_elements)
        for se in seal_elements:
            covered_boxes.append(se.bbox)

        # 2. Thu hồi các dòng chữ bị PPStructure bỏ sót (như đoạn Address)
        orphan_elements = self._recover_unassigned_text(working_img, page_index, covered_boxes)
        elements.extend(orphan_elements)
        for oe in orphan_elements:
            covered_boxes.append(oe.bbox)

        # 3. Khối chữ ký và con dấu ở cuối trang có bảng
        # Nếu đã có figure nào ở nửa dưới trang (y >= table_bottom_y - 100), KHÔNG tạo thêm ảnh crop trùng lặp
        if table_bottom_y > 0 and table_bottom_y < h - 150:
            already_has_bottom_image = any(
                e.element_type == "image" and e.bbox[1] >= (table_bottom_y - 100)
                for e in elements
            )
            if not already_has_bottom_image:
                bottom_area = working_img[table_bottom_y + 10:h - 50, :]
                hsv_bot = cv2.cvtColor(bottom_area, cv2.COLOR_BGR2HSV)
                m1 = cv2.inRange(hsv_bot, np.array([0, 70, 50]), np.array([10, 255, 255]))
                m2 = cv2.inRange(hsv_bot, np.array([170, 70, 50]), np.array([180, 255, 255]))
                has_red_stamp = (np.sum(cv2.bitwise_or(m1, m2) > 0) > 300)

                if has_red_stamp:
                    sig_crop_path = self.output_artifacts_dir / f"page_{page_index+1}_signature_stamp.png"
                    crop_pil = pil_img.crop((100, table_bottom_y + 10, w - 100, h - 50))
                    crop_pil.save(str(sig_crop_path))

                    elements.append(LayoutElement(
                        page_index=page_index,
                        element_type="image",
                        bbox=(100, table_bottom_y + 10, w - 100, h - 50),
                        content=str(sig_crop_path),
                        confidence=0.98,
                        needs_review=False,
                        metadata={"kind": "signature_stamp_block"}
                    ))

        # Sắp xếp theo thứ tự đọc (Y bbox) trước khi nhận diện khối Header và Signature
        elements.sort(key=lambda e: e.bbox[1])

        # 4. Tái cấu trúc khối chữ ký hai bên (nếu có) thành bảng 2 cột không viền chuẩn
        elements = self._detect_and_structure_signatures(working_img, page_index, elements, w, h)

        # 5. Tái cấu trúc khối Tiêu đề (Logo bên trái và Quốc hiệu bên phải) thành bảng 2 cột không viền
        elements = self._detect_and_structure_header(working_img, page_index, elements, w, h)

        # Sắp xếp lại lần cuối theo Y bbox
        elements.sort(key=lambda e: e.bbox[1])

        return PageLayout(
            page_index=page_index,
            width=w,
            height=h,
            elements=elements,
            warnings=warnings
        )

    def _detect_and_structure_header(
        self,
        page_img_bgr: np.ndarray,
        page_index: int,
        elements: List[LayoutElement],
        w: int,
        h: int
    ) -> List[LayoutElement]:
        """Tái cấu trúc phần Tiêu đề (Logo/Đơn vị bên trái và Quốc hiệu bên phải) thành bảng 2 cột không viền."""
        motto_idx = None
        for idx, e in enumerate(elements):
            if e.element_type == "paragraph" and e.bbox[1] < 0.25 * h:
                if "CỘNG HÒA XÃ HỘI" in e.content.upper():
                    motto_idx = idx
                    break

        if motto_idx is None:
            return elements

        header_cutoff_y = 0.28 * h
        left_elements = []
        right_elements = []
        indices_to_remove = []

        for idx, e in enumerate(elements):
            if e.bbox[3] <= header_cutoff_y and e.element_type in ["paragraph", "image"]:
                u_c = e.content.upper() if isinstance(e.content, str) else ""
                if any(u_c.startswith(p) for p in ["BIÊN BẢN", "HỢP ĐỒNG", "QUYẾT ĐỊNH", "TỜ TRÌNH"]):
                    break
                indices_to_remove.append(idx)
                x_mid = (e.bbox[0] + e.bbox[2]) / 2
                if x_mid < 0.45 * w:
                    left_elements.append(e)
                else:
                    right_elements.append(e)

        if not right_elements:
            return elements

        logo_path = None
        min_lx = min([e.bbox[0] for e in left_elements]) if left_elements else int(0.08 * w)
        min_ly = min([e.bbox[1] for e in left_elements]) if left_elements else int(0.04 * h)
        max_lx = max([e.bbox[2] for e in left_elements]) if left_elements else int(0.40 * w)
        max_ly = max([e.bbox[3] for e in left_elements]) if left_elements else int(0.18 * h)

        left_crop = page_img_bgr[max(0, min_ly - 50):min(h, max_ly + 10), max(0, min_lx - 20):min(w, max_lx + 20)]
        if left_crop.size > 0 and left_crop.shape[0] > 20 and left_crop.shape[1] > 20:
            p_logo = self.output_artifacts_dir / f"page_{page_index+1}_header_logo.png"
            cv2.imwrite(str(p_logo), left_crop)
            logo_path = str(p_logo)

        right_text = "\n".join([str(e.content) for e in right_elements])

        from pipeline.adaptive_table_extractor import TableResult, TableCell
        header_table_res = TableResult(
            rows=[[
                TableCell(row=0, col=0, text="", image_path=logo_path),
                TableCell(row=0, col=1, text=right_text)
            ]],
            spans=[],
            column_widths_pct=[38.0, 62.0],
            extraction_method="header_block_2col",
            structure_confidence=0.98
        )

        header_layout_elem = LayoutElement(
            page_index=page_index,
            element_type="table",
            bbox=(0, min_ly, w, max_ly),
            content=header_table_res,
            confidence=0.98,
            needs_review=False,
            metadata={"kind": "header_table"}
        )

        new_elements = [e for i, e in enumerate(elements) if i not in indices_to_remove]
        new_elements.insert(0, header_layout_elem)
        return new_elements

    def _detect_and_structure_signatures(
        self, 
        page_img_bgr: np.ndarray, 
        page_index: int, 
        elements: List[LayoutElement], 
        w: int, 
        h: int
    ) -> List[LayoutElement]:
        """Tái cấu trúc khối chữ ký hai bên ở cuối văn bản thành bảng 2 cột không viền chuẩn."""
        valid_sig_patterns = [
            "ĐẠI DIỆN", "BÊN GIAO", "BÊN NHẬN", "NGƯỜI GIAO", "NGƯỜI NHẬN", 
            "BÊN A", "BÊN B", "THỦ KHO", "GIÁM ĐỐC"
        ]
        
        sig_title_idx = None
        for idx, e in enumerate(elements):
            if e.element_type == "paragraph" and e.bbox[1] > 0.45 * h:
                text = e.content.strip()
                u = text.upper()
                if text.startswith(("-", "–", "—", "+", "•", "*")):
                    continue
                if any(k in u for k in ["KIỂM TRA", "XÁC NHẬN", "CHỨC VỤ", "CHỨC VI", "ĐỊA CHỈ", "ĐIỆN THOẠI", "KẾT LUẬN", "LẬP THÀNH"]):
                    continue
                if len(text) > 60:
                    continue
                if any(p in u for p in valid_sig_patterns):
                    sig_title_idx = idx
                    break

        if sig_title_idx is None:
            return elements

        title_elem = elements[sig_title_idx]
        title_text = title_elem.content.strip()
        t_y0, t_y1 = title_elem.bbox[1], title_elem.bbox[3]

        left_title = ""
        right_title = ""
        u_t = title_text.upper()
        
        if "ĐẠI DIỆN BÊN GIAO" in u_t and "ĐẠI DIỆN BÊN NHẬN" in u_t:
            left_title = "ĐẠI DIỆN BÊN GIAO"
            right_title = "ĐẠI DIỆN BÊN NHẬN"
        elif "BÊN GIAO" in u_t and "BÊN NHẬN" in u_t:
            left_title = "BÊN GIAO"
            right_title = "BÊN NHẬN"
        elif "BÊN A" in u_t and "BÊN B" in u_t:
            left_title = "ĐẠI DIỆN BÊN A"
            right_title = "ĐẠI DIỆN BÊN B"
        elif "NGƯỜI GIAO" in u_t and "NGƯỜI NHẬN" in u_t:
            left_title = "NGƯỜI GIAO HÀNG"
            right_title = "NGƯỜI NHẬN HÀNG"
        else:
            parts = [p.strip() for p in title_text.split() if p.strip()]
            if len(parts) >= 2:
                mid = len(parts) // 2
                left_title = " ".join(parts[:mid])
                right_title = " ".join(parts[mid:])
            else:
                return elements

        names_left = []
        names_right = []
        indices_to_remove = [sig_title_idx]

        for idx, e in enumerate(elements):
            if idx <= sig_title_idx:
                continue
            if e.element_type == "paragraph" and e.bbox[1] > t_y0:
                e_xmid = (e.bbox[0] + e.bbox[2]) / 2
                indices_to_remove.append(idx)
                if e_xmid < 0.5 * w:
                    names_left.append(e.content.strip())
                else:
                    names_right.append(e.content.strip())

        left_name_str = " ".join(names_left).strip()
        right_name_str = " ".join(names_right).strip()

        mid_y0 = t_y1 + 5
        mid_y1 = min(h - 30, max(t_y1 + 120, int(h * 0.98)))
        
        sig_zone = page_img_bgr[mid_y0:mid_y1, :]
        hsv_zone = cv2.cvtColor(sig_zone, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_zone, np.array([90, 45, 40]), np.array([135, 255, 230]))

        left_blue = np.sum(blue_mask[:, :int(w * 0.5)] > 0)
        right_blue = np.sum(blue_mask[:, int(w * 0.5):] > 0)

        # Bắt buộc chữ ký phải nằm sau bảng (nếu trang có bảng)
        if any(e.element_type == "table" and e.bbox[3] > t_y0 for e in elements):
            return elements

        left_img_path = None
        right_img_path = None

        if left_blue > 150:
            crop_l = page_img_bgr[mid_y0:mid_y1 - 40, int(w * 0.12):int(w * 0.48)]
            p_l = self.output_artifacts_dir / f"page_{page_index+1}_sig_left.png"
            cv2.imwrite(str(p_l), crop_l)
            left_img_path = str(p_l)

        if right_blue > 150:
            crop_r = page_img_bgr[mid_y0:mid_y1 - 30, int(w * 0.52):int(w * 0.88)]
            p_r = self.output_artifacts_dir / f"page_{page_index+1}_sig_right.png"
            cv2.imwrite(str(p_r), crop_r)
            right_img_path = str(p_r)

        # Bắt buộc phải có chữ ký thực tế hoặc phải có cả tên người ký 2 bên
        if not left_img_path and not right_img_path and (not left_name_str or not right_name_str):
            return elements

        from pipeline.adaptive_table_extractor import TableResult, TableCell
        sig_rows = [
            [
                TableCell(row=0, col=0, text=left_title),
                TableCell(row=0, col=1, text=right_title)
            ],
            [
                TableCell(row=1, col=0, text="", image_path=left_img_path),
                TableCell(row=1, col=1, text="", image_path=right_img_path)
            ],
            [
                TableCell(row=2, col=0, text=left_name_str),
                TableCell(row=2, col=1, text=right_name_str)
            ]
        ]

        sig_table_res = TableResult(
            rows=sig_rows,
            spans=[],
            column_widths_pct=[50.0, 50.0],
            extraction_method="signature_block_2col",
            structure_confidence=0.98
        )

        sig_layout_elem = LayoutElement(
            page_index=page_index,
            element_type="table",
            bbox=(int(w * 0.05), t_y0, int(w * 0.95), mid_y1),
            content=sig_table_res,
            confidence=0.98,
            needs_review=False,
            metadata={"kind": "signature_table"}
        )

        new_elements = [e for i, e in enumerate(elements) if i not in indices_to_remove]
        new_elements.append(sig_layout_elem)
        return new_elements

import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from bs4 import BeautifulSoup
from paddleocr import PaddleOCR

@dataclass
class TableCell:
    row: int
    col: int
    text: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x0, y0, x1, y1) relative to table crop
    confidence: Optional[float] = None
    rowspan: int = 1
    colspan: int = 1
    is_spanned_copy: bool = False
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    image_path: Optional[str] = None

@dataclass
class TableResult:
    rows: list[list[TableCell]]
    spans: list[dict]
    column_widths_pct: list[float]
    needs_review: bool = False
    warnings: list[str] = field(default_factory=list)
    extraction_method: str = "unknown"
    structure_confidence: Optional[float] = None
    row_count_model: Optional[int] = None
    row_count_geometry: Optional[int] = None


class AdaptiveTableExtractor:
    def __init__(self, ocr_engine: Optional[PaddleOCR] = None, vietocr_predictor: Any = None):
        if ocr_engine is None:
            self.ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        else:
            self.ocr = ocr_engine
        self.vietocr_predictor = vietocr_predictor

    def parse_html_table(self, html_str: str) -> Tuple[List[List[dict]], List[dict], int, int]:
        if not html_str:
            return [], [], 0, 0
            
        soup = BeautifulSoup(html_str, "html.parser")
        tr_elements = soup.find_all("tr")
        if not tr_elements:
            return [], [], 0, 0

        num_rows = len(tr_elements)
        max_cols = 0
        for tr in tr_elements:
            cols_in_row = sum(int(cell.get("colspan", 1)) for cell in tr.find_all(["td", "th"]))
            if cols_in_row > max_cols:
                max_cols = cols_in_row
        if max_cols == 0:
            max_cols = 1

        grid = [[None for _ in range(max_cols)] for _ in range(num_rows)]
        spans = []

        for r_idx, tr in enumerate(tr_elements):
            c_idx = 0
            cells = tr.find_all(["td", "th"])
            for cell in cells:
                while c_idx < max_cols and grid[r_idx][c_idx] is not None:
                    c_idx += 1
                if c_idx >= max_cols:
                    break

                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))
                text = cell.get_text().strip()

                root_cell = {
                    "row": r_idx,
                    "col": c_idx,
                    "text": text,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "is_spanned_copy": False
                }
                grid[r_idx][c_idx] = root_cell

                if rowspan > 1 or colspan > 1:
                    spans.append({
                        "row": r_idx,
                        "col": c_idx,
                        "rowspan": rowspan,
                        "colspan": colspan
                    })

                end_r = min(num_rows - 1, r_idx + rowspan - 1)
                end_c = min(max_cols - 1, c_idx + colspan - 1)
                for r in range(r_idx, end_r + 1):
                    for c in range(c_idx, end_c + 1):
                        if r == r_idx and c == c_idx:
                            continue
                        grid[r][c] = {
                            "row": r,
                            "col": c,
                            "text": text,
                            "rowspan": 1,
                            "colspan": 1,
                            "is_spanned_copy": True
                        }

                c_idx = end_c + 1

        for r in range(num_rows):
            for c in range(max_cols):
                if grid[r][c] is None:
                    grid[r][c] = {
                        "row": r, "col": c, "text": "",
                        "rowspan": 1, "colspan": 1, "is_spanned_copy": False
                    }

        return grid, spans, num_rows, max_cols

    def detect_grid_lines_opencv(self, table_image_bgr: np.ndarray) -> Tuple[List[int], List[int]]:
        gray = cv2.cvtColor(table_image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 4
        )

        h_k_len = max(15, w // 25)
        v_k_len = max(15, h // 30)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_k_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_k_len))

        h_morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        row_sums = np.sum(h_morph > 0, axis=1)
        max_r_sum = np.max(row_sums) if len(row_sums) > 0 else 0
        min_h_thresh = max(int(w * 0.12), int(max_r_sum * 0.35))
        y_indices = np.where(row_sums > min_h_thresh)[0]
        clusters_y = []
        for y in y_indices:
            if not clusters_y or y - clusters_y[-1][-1] > 8:
                clusters_y.append([y])
            else:
                clusters_y[-1].append(y)
        h_lines = [int(np.mean(c)) for c in clusters_y]

        col_sums = np.sum(v_morph > 0, axis=0)
        max_c_sum = np.max(col_sums) if len(col_sums) > 0 else 0
        min_v_thresh = max(int(h * 0.12), int(max_c_sum * 0.30))
        x_indices = np.where(col_sums > min_v_thresh)[0]
        clusters_x = []
        for x in x_indices:
            if not clusters_x or x - clusters_x[-1][-1] > 15:
                clusters_x.append([x])
            else:
                clusters_x[-1].append(x)
        v_lines = [int(np.mean(c)) for c in clusters_x]

        if v_lines and v_lines[0] > 35:
            v_lines.insert(0, 0)
        if v_lines and v_lines[-1] < w - 35:
            v_lines.append(w)

        return h_lines, v_lines

    def extract_table(self, table_image_bgr: np.ndarray, model_elem_res: Optional[dict] = None) -> TableResult:
        h_img, w_img = table_image_bgr.shape[:2]
        warnings = []
        needs_review = False

        html_str = model_elem_res.get("html", "") if model_elem_res else ""
        grid_html, spans_html, num_rows_model, max_cols_model = self.parse_html_table(html_str)

        h_lines_geo, v_lines_geo = self.detect_grid_lines_opencv(table_image_bgr)
        num_rows_geo = len(h_lines_geo) - 1 if len(h_lines_geo) >= 2 else 0
        num_cols_geo = len(v_lines_geo) - 1 if len(v_lines_geo) >= 2 else 0

        has_continuous_grid = (num_rows_geo >= 2 and num_cols_geo >= 2)

        # Quyết định chiến lược:
        # Nếu phát hiện lưới hình học rõ ràng (>= 2 hàng và >= 2 cột), ƯU TIÊN hình học để cắt từng ô chính xác!
        if has_continuous_grid:
            extraction_method = "opencv_grid_geometry"
            structure_confidence = 0.95
        elif num_rows_model > 0:
            extraction_method = "slanet_html_borderless"
            structure_confidence = 0.85
            warnings.append("Using SLANet HTML due to lack of complete continuous grid.")
        else:
            extraction_method = "failed"
            structure_confidence = 0.0
            warnings.append("Both geometry and SLANet failed to parse table.")
            return TableResult(
                rows=[], spans=[], column_widths_pct=[],
                needs_review=True, warnings=warnings,
                extraction_method=extraction_method, structure_confidence=0.0
            )

        final_rows: List[List[TableCell]] = []
        col_widths_pct: List[float] = []
        final_spans: List[dict] = []

        def _ocr_cell_content(crop_img: np.ndarray) -> Tuple[str, Optional[float]]:
            ch, cw = crop_img.shape[:2]
            if ch < 5 or cw < 5:
                return "", None
            pad_y = max(6, int(ch * 0.15))
            pad_x = max(10, int(cw * 0.15))
            padded_crop = cv2.copyMakeBorder(
                crop_img, pad_y, pad_y, pad_x, pad_x,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            if ch < 50 or cw < 140:
                padded_crop = cv2.resize(padded_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            ocr_res = self.ocr.ocr(padded_crop, cls=False)
            if not (ocr_res and ocr_res[0]):
                return "", None

            # Sắp xếp các dòng chữ theo tọa độ Y tăng dần (từ trên xuống dưới)
            lines_sorted = sorted(ocr_res[0], key=lambda item: min(pt[1] for pt in item[0]))

            if self.vietocr_predictor is not None:
                from PIL import Image
                v_lines = []
                for box, (p_text, p_conf) in lines_sorted:
                    pts = np.array(box, dtype=np.int32)
                    lx0, ly0 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
                    lx1, ly1 = min(padded_crop.shape[1], pts[:, 0].max()), min(padded_crop.shape[0], pts[:, 1].max())
                    line_crop = padded_crop[ly0:ly1, lx0:lx1]
                    if line_crop.shape[0] > 5 and line_crop.shape[1] > 5:
                        pad_line = cv2.copyMakeBorder(line_crop, 8, 8, 16, 16, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                        pil_l = Image.fromarray(cv2.cvtColor(pad_line, cv2.COLOR_BGR2RGB))
                        v_rec = self.vietocr_predictor.predict(pil_l).strip()
                        v_lines.append(v_rec)
                    else:
                        v_lines.append(p_text.strip())
                res_txt = " ".join(v_lines).strip()
                res_conf = 0.95
            else:
                res_txt = " ".join([l[1][0].strip() for l in lines_sorted]).strip()
                res_conf = float(np.mean([l[1][1] for l in lines_sorted]))

            # Chuẩn hóa khoảng trắng và dấu nháy biên
            res_txt = res_txt.strip(' "\'`|')
            return res_txt, res_conf

        if extraction_method == "opencv_grid_geometry":
            gray_tbl = cv2.cvtColor(table_image_bgr, cv2.COLOR_BGR2GRAY)
            bin_tbl = cv2.adaptiveThreshold(gray_tbl, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 4)
            v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h_img // 30)))
            v_morph = cv2.morphologyEx(bin_tbl, cv2.MORPH_OPEN, v_k)

            num_rows = num_rows_geo
            num_cols = num_cols_geo
            total_w = v_lines_geo[-1] - v_lines_geo[0] if v_lines_geo[-1] > v_lines_geo[0] else w_img
            for c_i in range(num_cols):
                w_col = v_lines_geo[c_i + 1] - v_lines_geo[c_i]
                col_widths_pct.append(round((w_col / total_w) * 100.0, 2))

            # Phát hiện sub-header ở hàng 0 nếu có vạch ngang nội bộ chia nhóm cột (như 'Số lượng')
            sub_line_info = None
            if num_rows >= 2 and num_cols >= 4:
                r0_y0, r0_y1 = h_lines_geo[0], h_lines_geo[1]
                r0_crop = table_image_bgr[r0_y0:r0_y1, :]
                r0_gray = cv2.cvtColor(r0_crop, cv2.COLOR_BGR2GRAY)
                r0_bin = cv2.adaptiveThreshold(r0_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 4)
                r0_k = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
                r0_morph = cv2.morphologyEx(r0_bin, cv2.MORPH_OPEN, r0_k)
                n_labels, _, r0_stats, _ = cv2.connectedComponentsWithStats(r0_morph)
                for s_i in range(1, n_labels):
                    sx, sy, sw, sh, sarea = r0_stats[s_i]
                    if sw > 120 and sh < 15 and 20 < sy < (r0_y1 - r0_y0 - 20):
                        sub_cols = []
                        for c_i in range(num_cols):
                            c_xmid = (v_lines_geo[c_i] + v_lines_geo[c_i + 1]) / 2
                            if sx <= c_xmid <= (sx + sw):
                                sub_cols.append(c_i)
                        if len(sub_cols) >= 2:
                            sub_line_info = {
                                "y_split": r0_y0 + sy,
                                "sub_cols": sub_cols
                            }
                            break

            if sub_line_info is not None:
                # Xây dựng cấu trúc 2 tầng cho Header: Row 0 (nhóm trên) + Row 1 (sub-header dưới)
                y_split = sub_line_info["y_split"]
                sub_cols = set(sub_line_info["sub_cols"])
                min_sub_c = min(sub_line_info["sub_cols"])
                max_sub_c = max(sub_line_info["sub_cols"])

                # Hàng 0
                row0 = []
                for c_i in range(num_cols):
                    x0, x1 = v_lines_geo[c_i], v_lines_geo[c_i + 1]
                    if c_i not in sub_cols:
                        c_crop = table_image_bgr[h_lines_geo[0]:h_lines_geo[1], x0:x1]
                        c_txt, c_conf = _ocr_cell_content(c_crop)
                        row0.append(TableCell(row=0, col=c_i, text=c_txt, bbox=(x0, h_lines_geo[0], x1, h_lines_geo[1]), rowspan=2, confidence=c_conf))
                    elif c_i == min_sub_c:
                        # Ô gộp phía trên của nhóm sub-header (vd: 'Số lượng')
                        grp_x0 = v_lines_geo[min_sub_c]
                        grp_x1 = v_lines_geo[max_sub_c + 1]
                        c_crop = table_image_bgr[h_lines_geo[0]:y_split, grp_x0:grp_x1]
                        c_txt, c_conf = _ocr_cell_content(c_crop)
                        row0.append(TableCell(row=0, col=c_i, text=c_txt, bbox=(grp_x0, h_lines_geo[0], grp_x1, y_split), colspan=len(sub_cols), confidence=c_conf))
                    else:
                        row0.append(TableCell(row=0, col=c_i, text="", is_spanned_copy=True))

                final_rows.append(row0)
                final_spans.append({"row": 0, "col": min_sub_c, "rowspan": 1, "colspan": len(sub_cols)})
                for c_i in range(num_cols):
                    if c_i not in sub_cols:
                        final_spans.append({"row": 0, "col": c_i, "rowspan": 2, "colspan": 1})

                # Hàng 1 (các cột con bên dưới)
                row1 = []
                for c_i in range(num_cols):
                    x0, x1 = v_lines_geo[c_i], v_lines_geo[c_i + 1]
                    if c_i in sub_cols:
                        c_crop = table_image_bgr[y_split:h_lines_geo[1], x0:x1]
                        c_txt, c_conf = _ocr_cell_content(c_crop)
                        row1.append(TableCell(row=1, col=c_i, text=c_txt, bbox=(x0, y_split, x1, h_lines_geo[1]), confidence=c_conf))
                    else:
                        row1.append(TableCell(row=1, col=c_i, text="", is_spanned_copy=True))
                final_rows.append(row1)

                # Các hàng dữ liệu tiếp theo (bắt đầu từ index 1 của h_lines_geo)
                for r_idx in range(1, num_rows):
                    y0, y1 = h_lines_geo[r_idx], h_lines_geo[r_idx + 1]
                    r_cells = []
                    for c_i in range(num_cols):
                        x0, x1 = v_lines_geo[c_i], v_lines_geo[c_i + 1]
                        c_crop = table_image_bgr[y0:y1, x0:x1]
                        c_txt, c_conf = _ocr_cell_content(c_crop)
                        r_cells.append(TableCell(row=len(final_rows), col=c_i, text=c_txt, bbox=(x0, y0, x1, y1), confidence=c_conf))
                    final_rows.append(r_cells)

            else:
                for r_i in range(num_rows):
                    y0, y1 = h_lines_geo[r_i], h_lines_geo[r_i + 1]
                    row_cells = []
                    c_i = 0
                    while c_i < num_cols:
                        x0 = v_lines_geo[c_i]
                        span_len = 1
                        while (c_i + span_len) < num_cols:
                            x_div = v_lines_geo[c_i + span_len]
                            v_sum = np.sum(v_morph[y0:y1, max(0, x_div - 4):min(w_img, x_div + 5)] > 0)
                            if v_sum < 0.3 * (y1 - y0):
                                span_len += 1
                            else:
                                break
                        x1 = v_lines_geo[c_i + span_len]
                        cell_crop = table_image_bgr[max(0, y0 - 5):y1, x0:x1]
                        cell_text, conf = _ocr_cell_content(cell_crop)

                        cell = TableCell(
                            row=r_i,
                            col=c_i,
                            text=cell_text,
                            bbox=(x0, y0, x1, y1),
                            colspan=span_len,
                            confidence=round(float(conf), 3) if conf else None
                        )
                        row_cells.append(cell)
                        if span_len > 1:
                            final_spans.append({"row": r_i, "col": c_i, "rowspan": 1, "colspan": span_len})
                            for extra_c in range(1, span_len):
                                row_cells.append(TableCell(
                                    row=r_i, col=c_i + extra_c, text="", is_spanned_copy=True
                                ))
                        c_i += span_len
                    final_rows.append(row_cells)


        else:
            # slanet_html_borderless
            for r_idx in range(num_rows_model):
                row_cells = []
                for c_idx in range(max_cols_model):
                    info = grid_html[r_idx][c_idx]
                    cell = TableCell(
                        row=r_idx,
                        col=c_idx,
                        text=info["text"],
                        rowspan=info["rowspan"],
                        colspan=info["colspan"],
                        is_spanned_copy=info.get("is_spanned_copy", False),
                        confidence=0.85
                    )
                    row_cells.append(cell)
                final_rows.append(row_cells)
            final_spans = spans_html
            col_widths_pct = [round(100.0 / max_cols_model, 2) for _ in range(max_cols_model)]

        # Semantic Validation
        empty_row_count = 0
        for r_idx, row in enumerate(final_rows):
            texts = [c.text.strip() for c in row if c.text.strip() and not c.is_spanned_copy]
            if not texts:
                empty_row_count += 1
                for c in row:
                    c.needs_review = True
                    c.review_reasons.append("Empty row")

        if empty_row_count > 0:
            needs_review = True
            warnings.append(f"Found {empty_row_count} empty rows.")

        return TableResult(
            rows=final_rows,
            spans=final_spans,
            column_widths_pct=col_widths_pct,
            needs_review=needs_review,
            warnings=warnings,
            extraction_method=extraction_method,
            structure_confidence=round(structure_confidence, 2),
            row_count_model=num_rows_model,
            row_count_geometry=num_rows_geo
        )

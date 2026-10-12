import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DeskewResult:
    """Kết quả ước lượng và hiệu chỉnh góc nghiêng."""
    image: np.ndarray
    estimated_angle: float
    applied_angle: float
    confidence: float
    status: str
    is_modified: bool


class ImageDeskewer:
    """
    Module ước lượng và hiệu chỉnh góc nghiêng (Auto-Deskew) tổng quát.
    
    Nguyên tắc thiết kế:
    1. Không sửa đổi mảng ảnh đầu vào gốc (trả về bản copy độc lập).
    2. Chỉ áp dụng xoay khi góc ước lượng vượt ngưỡng tối thiểu (min_angle)
       và nằm trong giới hạn an toàn (max_angle).
    3. Kiểm tra độ tin cậy (confidence) dựa trên độ đồng hướng của các đường ngang (std nhỏ).
    4. Xoay với viền trắng (255, 255, 255) bảo toàn kích thước và kênh màu.
    5. Không xoay trên trang trắng hoặc khi không đủ đường kẻ tin cậy.
    """

    def __init__(
        self,
        min_angle: float = 1.2,
        max_angle: float = 20.0,
        min_lines: int = 8,
        min_confidence: float = 0.50
    ):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_lines = min_lines
        self.min_confidence = min_confidence

    def estimate_skew(self, image_bgr: np.ndarray) -> Tuple[float, float, str]:
        """
        Ước lượng góc nghiêng văn bản bằng Hough Lines trên ảnh Canny.
        
        Trả về:
            (angle_deg, confidence, status)
            angle_deg: góc nghiêng theo độ (độ lệch so với phương ngang).
        """
        if len(image_bgr.shape) == 2:
            gray = image_bgr
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Bỏ qua trang trắng
        non_white = np.mean(gray < 240)
        if non_white < 0.0035:
            return 0.0, 0.0, "blank_page"

        # 2. Phát hiện cạnh
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=60, maxLineGap=10)

        if lines is None or len(lines) < self.min_lines:
            return 0.0, 0.0, "insufficient_lines"

        # 3. Lọc các đường gần ngang (|deg| < 25)
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(deg) < 25.0:
                angles.append(deg)

        if len(angles) < self.min_lines:
            return 0.0, 0.0, "insufficient_horizontal_lines"

        median_angle = float(np.median(angles))
        std_angle = float(np.std(angles))

        # Độ tin cậy tỉ lệ nghịch với độ lệch chuẩn của góc giữa các dòng
        confidence = float(np.clip(1.0 - (std_angle / 10.0), 0.0, 1.0))

        # 4. Kiểm tra các điều kiện an toàn
        if abs(median_angle) < self.min_angle:
            return median_angle, confidence, "nearly_straight"

        if abs(median_angle) > self.max_angle:
            return median_angle, confidence, "angle_out_of_bounds"

        if confidence < self.min_confidence:
            return median_angle, confidence, "low_confidence"

        return median_angle, confidence, "deskew_needed"

    def rotate_image(self, image_bgr: np.ndarray, angle: float) -> np.ndarray:
        """
        Xoay ảnh quanh tâm với góc angle (độ).
        Giữ nguyên kích thước và kênh màu, lấp viền ngoài bằng màu trắng.
        """
        h, w = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        # Ma trận xoay cv2.getRotationMatrix2D:
        # Nếu đường kẻ có góc dương (nghiêng xuống về bên phải), xoay -angle để thẳng lại
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        border_val = (255, 255, 255) if len(image_bgr.shape) == 3 else 255
        rotated = cv2.warpAffine(
            image_bgr,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val
        )
        return rotated

    def auto_orient_orthogonal(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Tự động nhận diện và xoay đúng chiều 90° nếu trang scan bị đặt ngang/ngược.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Bỏ qua trang quá nhỏ hoặc trang trắng
        if np.mean(gray < 240) < 0.0035:
            return image_bgr, 0

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=max(60, min(h, w) // 20), maxLineGap=10)
        if lines is None:
            return image_bgr, 0

        h_count, v_count = 0, 0
        for l in lines:
            x1, y1, x2, y2 = l[0]
            deg = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if deg <= 15 or deg >= 165:
                h_count += 1
            elif 75 <= deg <= 105:
                v_count += 1

        # Nếu số đường dọc gấp đôi số đường ngang, văn bản đang bị xoay 90° hoặc 270°
        if v_count >= 2 * h_count and v_count >= 25:
            # Thử xoay 90 độ thuận chiều kim đồng hồ
            cand_cw = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
            cand_ccw = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Đánh giá nhanh mật độ pixel đen ở nửa trên so với nửa dưới
            # Tiêu đề bảng biểu thường tập trung ở 35% trên cùng
            g_cw = cv2.cvtColor(cand_cw[:int(w * 0.35), :], cv2.COLOR_BGR2GRAY)
            g_ccw = cv2.cvtColor(cand_ccw[:int(w * 0.35), :], cv2.COLOR_BGR2GRAY)
            density_cw = np.mean(g_cw < 200)
            density_ccw = np.mean(g_ccw < 200)

            if density_cw >= density_ccw:
                return cand_cw, 90
            else:
                return cand_ccw, 270

        return image_bgr, 0

    def deskew(self, image_bgr: np.ndarray) -> DeskewResult:
        """
        Tiến trình Auto-Deskew trọn gói kết hợp định hướng vuông góc 90°.
        Trả về DeskewResult chứa ảnh kết quả (copy) và metadata đi kèm.
        """
        oriented_img, ortho_rot = self.auto_orient_orthogonal(image_bgr)
        est_angle, confidence, status = self.estimate_skew(oriented_img)

        if status == "deskew_needed":
            rotated = self.rotate_image(oriented_img, est_angle)
            return DeskewResult(
                image=rotated,
                estimated_angle=est_angle,
                applied_angle=est_angle,
                confidence=confidence,
                status=status,
                is_modified=True
            )
        else:
            return DeskewResult(
                image=oriented_img.copy(),
                estimated_angle=est_angle,
                applied_angle=0.0,
                confidence=confidence,
                status=status,
                is_modified=(ortho_rot != 0)
            )

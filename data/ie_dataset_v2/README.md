# Cấu trúc & Quy trình IE Dataset V2

Đây là không gian lưu trữ cho Giai đoạn B của dự án. 
Mục tiêu: Đánh giá độ phủ và độ chính xác của Evaluator trên 20-30 chứng từ đa dạng.

## 1. Cấu trúc thư mục

*   `raw/` — Chứng từ gốc (PDF, ảnh). **Tuyệt đối không sửa hoặc ghi đè.** (Ví dụ đặt tên: `DOC-0001.pdf`).
*   `annotations/` — Ground Truth JSON theo tài liệu. (Mỗi file JSON tương ứng 1 Document ID, ví dụ `DOC-0001.json`, có liên kết `source_file` tới `raw/`). Xem `schema_template.json`.
*   `splits/` — Danh sách chia tập (train/dev/test). Tài liệu trùng hoặc gần trùng phải được gộp chung vào cùng một tập, không được để rò rỉ (leak) dữ liệu giữa các tập.
*   `metadata/` — Thông tin bổ trợ (Loại chứng từ, nguồn, số trang).

## 2. Nguyên tắc Ground Truth & Trạng Thái

1.  **Độc lập OCR:** Không đặt kết quả OCR vào trường Ground Truth. Giá trị `expected_value` là nhãn thực tế đọc bằng mắt người trên bản scan gốc.
2.  **Tách bạch Vai trò (Roles):** Không gộp chung mọi pháp nhân vào một trường `company_name`. Phải tách rõ `recipient_company` và `issuer_company`.
3.  **Quy ước Vị trí (Bounding Box):** Nếu có, lưu theo định dạng `[x1, y1, x2, y2]` (Đơn vị: Pixel). Luôn ghi nhận `page` tham chiếu. Trả về `null` nếu chưa xác minh.
4.  **Trạng thái Tồn tại (Presence) & Xác minh (Verification):**
    *   `verification_status`: Chỉ đánh giá metric cho các trường `CONFIRMED`. Trạng thái `PENDING` sẽ bị skip.
    *   `presence`: Thể hiện sự tồn tại của trường trên giấy:
        *   `PRESENT`: Trường có xuất hiện và đọc được.
        *   `ABSENT`: Trường thực sự không xuất hiện (Đánh giá thuật toán có bắt nhầm hay không). Không dùng chuỗi rỗng thông thường.
        *   `UNREADABLE`: Có trường nhưng mờ/rách không đọc được.

## 3. Quy tắc Chuẩn hóa (Normalization Rules)
*   **Khoảng trắng (Whitespace):** Xóa khoảng trắng thừa ở đầu/cuối. Khoảng trắng giữa các từ chỉ giữ lại 1 khoảng (Ví dụ: `A   B` -> `A B`).
*   **Chữ Hoa/Thường (Case):** Phụ thuộc vào nghiệp vụ cụ thể. Mặc định sẽ dùng Exact Match (có phân biệt hoa/thường) trừ khi Evaluator có chỉ thị normalize.
*   **Dấu câu (Punctuation):** Giữ nguyên dấu câu của văn bản gốc (Dấu phẩy, ngoặc đơn). 

## 4. Xử lý Evaluator
*   Evaluator tương lai cần xử lý riêng trường `presence == "ABSENT"`: Nếu mô hình trả về giá trị (thay vì rỗng) cho trường ABSENT -> Đó là `False Positive`.
*   Evaluator v2 không được ghi đè `artifacts/ie_evaluation/frozen_baseline_3_cases_v1.json` của hệ thống cũ.

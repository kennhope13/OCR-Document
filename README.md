# BE-OCR Core: End-to-End Document Reconstruction & Intelligence Pipeline

Dự án xử lý OCR nâng cao, phân tích cấu trúc trang văn bản (Layout Analysis), trích xuất thực thể thông tin (Information Extraction) và **tái tạo trung thực 1:1 văn bản quét (Scanned Document) sang định dạng Microsoft Word (.docx)** có thể chỉnh sửa trực tiếp.

---

## 🌟 Tính Năng Nổi Bật

### 1. Tái Tạo Văn Bản Quét Sang Word 100% Trung Thực (Pixel-Level to Native DOCX)
- **Tự động khử nghiêng (Adaptive Deskewing)**: Khôi phục văn bản bị lệch xoay từ $\pm 1.2^\circ$ đến $20^\circ$.
- **Tái tạo bảng biểu thích ứng (Adaptive Table Extraction)**:
  - Tự động phát hiện viền bảng mờ, viền đứt nét bằng hình thái học thích ứng (Morphological kernels tỷ lệ động).
  - Tự động nhận diện gộp ô ngang/dọc (`colspan`, `rowspan`), phân tách ma trận ô dữ liệu phức tạp (như danh sách linh kiện, số serial máy tính).
- **Phân tách Run đa sắc & kiểu dáng (Multi-Run Styling)**:
  - Nhận diện màu mực bút bi xanh (HSV blue mask) và phân tách trong từng dòng: Nhãn trường in sẵn mang **chữ đen chuẩn**, chữ viết tay điền thêm mang **chữ xanh thực tế**.
  - Tự động nhận diện chữ gạch chân (Underline), in đậm (Bold), in nghiêng (Italic), cỡ chữ linh hoạt theo DPI ảnh scan.
- **Khử nhiễu hàng chấm (Dotted Fill-in Line Cleanup)**:
  - Loại bỏ hiện tượng nét bút bi đè lên hàng chấm in sẵn gây nhận dạng sai chữ cái (`: khạm. Minh... Tiên` $\rightarrow$ `Phạm Minh Tiên`).
- **Khớp thực thể danh tính chéo trang (Cross-Entity Co-Reference)**:
  - Khôi phục chữ viết tay mờ/nhiễu dựa trên đối soát thực thể với chữ ký đại diện hợp lệ ở cuối tài liệu (hoàn toàn không hardcode).
- **Dựng Layout hành chính chuyên nghiệp**:
  - Tiêu đề Quốc hiệu & Logo 2 cột không viền (Header 2-column borderless layout).
  - Khối chữ ký song song 2 bên (Đại diện bên giao - Bên nhận) tích hợp ảnh crop chữ ký thực tế.

### 2. Trích Xuất & Hiệu Đính Thực Thể Thông Tin (IE & Correction Layer)
- Trích xuất thông tin hợp đồng, số hóa đơn, số C/O, C/Q, thông tin bên giao/nhận, danh mục hàng hóa.
- Lớp chuẩn hóa tên hàng hóa (Commodity Normalizer) và luật sửa lỗi chính tả OCR (Correction Rules).

### 3. Giao Diện Kiểm Thử Trực Quan (Streamlit UI Dashboard)
- Cho phép tải tài liệu PDF/ảnh quét lên và xem so sánh song song giữa ảnh quét gốc và tài liệu Word được sinh ra theo từng trang.
- Hỗ trợ tải về file `.docx` hoàn chỉnh.

---

## 📁 Cấu Trúc Dự Án

```text
be-ocr-core/
├── app.py                      # Flask API Service phục vụ tích hợp
├── ui_test_dashboard.py        # Giao diện Streamlit tương tác kiểm thử tài liệu
├── pipeline/
│   ├── document_reconstructor.py   # Lõi phân tích bố cục, khử nhiễu & phân loại phần tử
│   ├── adaptive_table_extractor.py # Thuật toán trích xuất bảng & gộp ô ma trận thích ứng
│   ├── docx_builder.py             # Dựng file Microsoft Word (.docx) chuẩn cấu trúc & định dạng
│   ├── deskew.py                   # Thuật toán khử xoay nghiêng ảnh
│   └── language_detector.py        # Nhận diện ngôn ngữ văn bản
├── ie/                         # Lớp trích xuất thông tin (Information Extraction)
│   ├── schema.py               # Pydantic schemas cho dữ liệu trích xuất
│   └── correction/             # Lớp chuẩn hóa và sửa lỗi hậu OCR
├── ocr/                        # OCR handlers (VietOCR, PaddleOCR)
├── tests/                      # Bộ kiểm thử đơn vị & hồi quy (Unit & Regression Tests)
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt

### 1. Yêu Cầu Môi Trường
- Python >= 3.10
- Windows / Linux / macOS
- Đã cài đặt Git

### 2. Cài Đặt Thư Viện

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1
# (Hoặc trên Linux/macOS: source .venv/bin/activate)

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

---

## 💻 Hướng Dẫn Sử Dụng

### 1. Khởi Chạy Giao Diện Kiểm Thử Streamlit
Để xem trước và tái tạo trực quan các tài liệu scan phức tạp:
```bash
streamlit run ui_test_dashboard.py
```
Truy cập trình duyệt tại `http://localhost:8501`, tải lên file PDF (ví dụ Biên bản bàn giao, Invoice, Hợp đồng) và bấm **Bắt đầu Tái tạo Document**.

### 2. Khởi Chạy API Service
```bash
python app.py
```
Dịch vụ chạy tại `http://localhost:5000`.

---

## 🧪 Chạy Kiểm Thử (Tests)

Chạy bộ test tự động để đảm bảo độ chính xác của pipeline:
```bash
pytest
```
Hoặc chạy kiểm thử riêng module tái tạo bảng và tài liệu:
```bash
pytest tests/pipeline/test_docx_builder.py
pytest tests/pipeline/test_table_parser.py
```

---

## 📜 Bản Quyền
Dự án được phát triển phục vụ công tác số hóa, xử lý văn bản hành chính và tài liệu hải quan tự động.
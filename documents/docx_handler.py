from docx import Document
from docx.shared import Pt, Inches
from docx.enum.section import WD_ORIENT

# Tạo docx
def create_docx(ocr_result,file_name):
    min_width = float('inf')
    doc = Document()
    for item in ocr_result[0]:
        coordinates = item[0]  
        text = item[1]  
        text_values = text[0]
        x1, y1 = coordinates[0][0], coordinates[0][1]  # Góc trên trái

        # Tạo các đoạn văn bản và thêm vào file DOCX
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(text_values)
        
        # Xử lý chiều rộng và chiều cao của văn bản để định dạng
        width = coordinates[1][0] - coordinates[0][0]
        height = coordinates[2][1] - coordinates[0][1]
        font_size = min(width, height)
        run.font.size = Pt(font_size)

        # Đặt các thuộc tính của đoạn văn bản
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.left_indent = Pt(x1 )
        paragraph.paragraph_format.space_before =  Pt(y1 *0.000001)
        if x1 < min_width:
            min_width = x1
        # paragraph.paragraph_format.right_indent = Pt(x1)
        section = doc.sections[-1]
        section.orientation = WD_ORIENT.LANDSCAPE
        
        section.page_width = Inches(min_width)

    output_path = file_name.replace(".jpg", ".docx").replace(".jpeg", ".docx").replace(".png", ".docx")
    doc.save(output_path)
    return output_path
    

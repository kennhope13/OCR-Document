from flask import Flask, request, jsonify, send_file,after_this_request
from ocr.ocr_handler import process_pdf, process_image
from preprocessing_image.preprocessing_image import deskew
from PIL import Image
import os
import io
import traceback
from documents.docx_handler import create_docx
from google.cloud import storage
import threading
import numpy as np
import cv2
app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert_ocr():
    try:
        # Kết nối với Google Cloud Storage
        client = storage.Client()
        bucket_name = "ocr_project_file_storage"
        bucket = client.get_bucket(bucket_name)

        if not bucket:
            return jsonify({"error": "Failed to connect to the bucket."}), 500

        # print(f"Connected to bucket: {bucket_name}")

        # Kiểm tra thông tin file_name từ request
        file_name = request.form.get('file_name')
        if not file_name:
            return jsonify({"error": "No file_name provided."}), 400
        file_path = f"input/image_to_word/{file_name}"

        # Kiểm tra file trong bucket
        blob = bucket.blob(file_path)
        if not blob.exists():
            return jsonify({"error": f"File '{file_name}' does not exist in bucket."}), 404
        
        # Tải file vào bộ nhớ
        file_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(file_bytes))
        image_np = np.array(image)

        # Xử lý ảnh nghiêng
        pre_file_img = deskew(image_np)

        # Lấy định dạng từ tên file
        image_format = file_name.split('.')[-1].lower()
        if image_format not in ['jpg', 'jpeg', 'png']:
            return jsonify({"error": "Unsupported image format."}), 400
        
        # Ánh xạ định dạng file chuẩn trong pillow
        pillow_format = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG"
        }
        file_stream = io.BytesIO()
        Image.fromarray(cv2.cvtColor(pre_file_img, cv2.COLOR_BGR2RGB)).save(file_stream, format=pillow_format[image_format]) 
        file_stream.seek(0)

        # Xử lý OCR dựa trên loại file
        if file_name.lower().endswith('.pdf'):
            ocr_result = process_pdf(file_stream, file_name)
        elif file_name.lower().endswith(('jpg', 'jpeg', 'png')):
            ocr_result = process_image(file_stream, file_name)
        else:
            return jsonify({"error": "Unsupported file format."}), 400

        # Tạo file DOCX từ kết quả OCR
        if ocr_result:
            try:
                docx_file = create_docx(ocr_result, file_name)
                blob_path_output = f"output/image_to_word/{docx_file}"
                blob = bucket.blob(blob_path_output)
                blob.upload_from_filename(docx_file)
                os.remove(docx_file)
                print(f"File '{blob_path_output}' has been upload successfully!")
            except ValueError as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "OCR result is empty or invalid."}), 400
        
        return jsonify({
            "message": "OCR process completed",
            "docx_file": docx_file
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    

@app.route('/<filename>', methods=['GET'])
def download_file(filename):
    file_path = filename  # Giả sử tệp đã được lưu trong thư mục hiện tại

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    @after_this_request
    def delete_file(response):
        threading.Thread(target=remove_file_later, args=(file_path,)).start()  # Xóa trong luồng riêng biệt
        return response

    return send_file(file_path, as_attachment=True)  # Gửi tệp cho người dùng

def remove_file_later(file_path):
    try:
        os.remove(file_path)  # Xóa tệp sau khi phản hồi được gửi
        print(f"File {file_path} deleted successfully.")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
    
if __name__ == '__main__':
    app.run(debug=True)
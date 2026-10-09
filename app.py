import io
import os
import traceback

from flask import Flask, after_this_request, jsonify, request, send_file
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage
from PIL import Image
import numpy as np

from documents.docx_handler import create_docx
from ocr.ocr_handler import detect_file_kind, is_supported_file, process_image, process_pdf
from preprocessing_image.preprocessing_image import deskew

app = Flask(__name__)


def get_storage_client():
    project_id = os.getenv('GCS_PROJECT_ID', 'test-project')
    if os.getenv('STORAGE_EMULATOR_HOST'):
        return storage.Client(project=project_id, credentials=AnonymousCredentials())
    return storage.Client(project=project_id)


def get_bucket_name():
    return os.getenv('GCS_BUCKET_NAME', 'ocr_project_file_storage')


def ensure_bucket_exists(bucket):
    try:
        if not bucket.exists():
            bucket.create()
    except Exception:
        if not bucket.exists():
            raise


@app.route('/convert', methods=['POST'])
def convert_ocr():
    try:
        file_name = request.form.get('file_name')
        if not file_name:
            return jsonify({"error": "No file_name provided."}), 400

        file_bytes = request.files.get('file')
        if file_bytes is None:
            return jsonify({"error": "No file uploaded."}), 400

        file_bytes = file_bytes.read()
        file_kind = detect_file_kind(file_name, file_bytes)
        if not is_supported_file(file_name, file_bytes):
            return jsonify({"error": "Unsupported file format."}), 400

        if file_kind == 'image':
            image = Image.open(io.BytesIO(file_bytes))
            image_np = np.array(image)
            pre_file_img = deskew(image_np)

            image_format = file_name.split('.')[-1].lower()
            pillow_format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}
            if image_format not in pillow_format:
                return jsonify({"error": "Unsupported image format."}), 400

            file_stream = io.BytesIO()
            Image.fromarray(pre_file_img).save(file_stream, format=pillow_format[image_format])
            file_stream.seek(0)
            ocr_result = process_image(file_stream, file_name)
        elif file_kind == 'pdf':
            ocr_result = process_pdf(file_bytes, file_name=file_name)
        else:
            return jsonify({"error": "Unsupported file format."}), 400

        if not ocr_result:
            return jsonify({"error": "OCR result is empty or invalid."}), 400

        docx_file = create_docx(ocr_result, file_name)

        client = get_storage_client()
        bucket_name = get_bucket_name()
        bucket = client.bucket(bucket_name)
        if os.getenv("STORAGE_EMULATOR_HOST"):
            ensure_bucket_exists(bucket)
        blob_path_output = f"output/image_to_word/{os.path.basename(docx_file)}"
        blob = bucket.blob(blob_path_output)
        blob.upload_from_filename(docx_file)

        if os.path.exists(docx_file):
            os.remove(docx_file)

        return jsonify({
            "message": "OCR process completed",
            "docx_file": os.path.basename(docx_file),
            "download_url": f"/download/{os.path.basename(docx_file)}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    local_path = filename
    if os.path.exists(local_path):
        return send_file(local_path, as_attachment=True)

    try:
        client = get_storage_client()
        bucket_name = get_bucket_name()
        bucket = client.bucket(bucket_name)
        blob_path = f"output/image_to_word/{filename}"
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return jsonify({"error": "File not found"}), 404

        file_bytes = blob.download_as_bytes()
        return send_file(io.BytesIO(file_bytes), as_attachment=True, download_name=filename)
    except Exception:
        return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    app.run(debug=False)
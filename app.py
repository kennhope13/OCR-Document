import io
import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory, send_file
from werkzeug.utils import secure_filename
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage

from pipeline.unified_engine import UnifiedDocumentEngine, SecurityValidationError, DocumentProcessingError
from ocr.ocr_handler import detect_file_kind, is_supported_file

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Security Configurations
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB Max Upload Limit
BASE_DIR = Path(__file__).resolve().parent
LOCAL_OUTPUT_DIR = BASE_DIR / "output" / "image_to_word"
LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared Unified Engine instance
_engine: Optional[UnifiedDocumentEngine] = None


def get_engine() -> UnifiedDocumentEngine:
    global _engine
    if _engine is None:
        _engine = UnifiedDocumentEngine(
            lang=os.getenv("OCR_LANG", "vi"),
            artifacts_dir=str(BASE_DIR / "artifacts" / "reconstruct")
        )
    return _engine


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
        raw_file_name = request.form.get('file_name', '')
        file_obj = request.files.get('file')
        if file_obj is None:
            return jsonify({"error": "No file uploaded."}), 400

        # Read content safely
        file_bytes = file_obj.read()
        if not file_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # Validate file kind and extension
        file_kind = detect_file_kind(raw_file_name, file_bytes)
        if not is_supported_file(raw_file_name, file_bytes):
            return jsonify({"error": "Unsupported file format."}), 400

        # Generate a safe, collision-free server-side filename
        safe_base_name = secure_filename(Path(raw_file_name).stem) or "document"
        unique_file_id = f"{safe_base_name}_{uuid.uuid4().hex[:8]}.docx"
        target_docx_path = LOCAL_OUTPUT_DIR / unique_file_id

        # Execute unified processing pipeline
        engine = get_engine()
        saved_docx, page_layouts = engine.process_file_bytes(
            file_bytes=file_bytes,
            file_kind=file_kind,
            output_docx_path=target_docx_path,
        )

        # Upload to Google Cloud Storage if configured / required
        try:
            client = get_storage_client()
            bucket_name = get_bucket_name()
            bucket = client.bucket(bucket_name)
            if os.getenv("STORAGE_EMULATOR_HOST"):
                ensure_bucket_exists(bucket)
            blob_path_output = f"output/image_to_word/{unique_file_id}"
            blob = bucket.blob(blob_path_output)
            blob.upload_from_filename(str(saved_docx))
        except Exception as gcs_err:
            logger.warning(f"GCS upload skipped or failed: {gcs_err}")

        return jsonify({
            "message": "OCR process completed successfully",
            "docx_file": unique_file_id,
            "total_pages": len(page_layouts),
            "download_url": f"/download/{unique_file_id}"
        }), 200

    except SecurityValidationError as sve:
        return jsonify({"error": f"Security validation failed: {str(sve)}"}), 400
    except DocumentProcessingError as dpe:
        return jsonify({"error": f"Document processing failed: {str(dpe)}"}), 500
    except Exception as e:
        logger.exception("Unexpected error in /convert")
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Safely send file from strictly restricted output directory to prevent path traversal."""
    safe_name = secure_filename(filename)
    if not safe_name or not safe_name.endswith('.docx'):
        return jsonify({"error": "Invalid file requested."}), 400

    local_file_path = LOCAL_OUTPUT_DIR / safe_name
    if local_file_path.is_file():
        return send_from_directory(
            str(LOCAL_OUTPUT_DIR),
            safe_name,
            as_attachment=True,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    # Fallback to GCS storage if not locally present
    try:
        client = get_storage_client()
        bucket_name = get_bucket_name()
        bucket = client.bucket(bucket_name)
        blob_path = f"output/image_to_word/{safe_name}"
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return jsonify({"error": "File not found"}), 404

        file_bytes = blob.download_as_bytes()
        return send_file(
            io.BytesIO(file_bytes),
            as_attachment=True,
            download_name=safe_name,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception:
        return jsonify({"error": "File not found"}), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File exceeds maximum allowed upload size (32MB)."}), 413


if __name__ == '__main__':
    app.run(debug=False)
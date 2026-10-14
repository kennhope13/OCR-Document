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

# Security & Storage Configurations
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB Max Upload Limit
BASE_DIR = Path(__file__).resolve().parent
LOCAL_OUTPUT_DIR = BASE_DIR / "output" / "image_to_word"
LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import hmac
import hashlib
import time

# Retention / TTL Configuration for local outputs (in seconds, default 2 hours)
OUTPUT_RETENTION_SECONDS = int(os.getenv("OUTPUT_RETENTION_SECONDS", 7200))

# HMAC Secret & Token Expiration for Download URLs (default token validity: 1800s / 30 mins)
DOWNLOAD_SECRET_KEY = os.getenv("DOWNLOAD_SECRET_KEY", "ocr-secret-download-key-default").encode("utf-8")
DOWNLOAD_TOKEN_TTL_SECONDS = int(os.getenv("DOWNLOAD_TOKEN_TTL_SECONDS", 1800))


def generate_download_token(filename: str, expires_at: int) -> str:
    """Generate secure HMAC-SHA256 signature for filename and expiration timestamp."""
    payload = f"{filename}:{expires_at}".encode("utf-8")
    return hmac.new(DOWNLOAD_SECRET_KEY, payload, hashlib.sha256).hexdigest()


def verify_download_token(filename: str, expires_at: int, token: str) -> bool:
    """Verify HMAC signature with constant-time comparison and check expiration timestamp."""
    if not token or time.time() > expires_at:
        return False
    expected = generate_download_token(filename, expires_at)
    return hmac.compare_digest(expected, token)


import threading


# Shared Unified Engine instance with thread-safe singleton lock
_engine: Optional[UnifiedDocumentEngine] = None
_engine_init_lock = threading.Lock()


def get_engine() -> UnifiedDocumentEngine:
    global _engine
    if _engine is None:
        with _engine_init_lock:
            if _engine is None:
                _engine = UnifiedDocumentEngine(
                    lang=os.getenv("OCR_LANG", "vi"),
                    artifacts_dir=str(BASE_DIR / "artifacts" / "reconstruct")
                )
    return _engine


def cleanup_expired_output_files(max_age_seconds: int = OUTPUT_RETENTION_SECONDS) -> int:
    """Removes output files older than max_age_seconds from LOCAL_OUTPUT_DIR."""
    import time
    now = time.time()
    deleted_count = 0
    try:
        for f in LOCAL_OUTPUT_DIR.glob("*.docx"):
            if f.is_file():
                file_age = now - f.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        f.unlink()
                        deleted_count += 1
                    except Exception as err:
                        logger.warning(f"Failed to delete expired file {f}: {err}")
    except Exception as e:
        logger.warning(f"Error during output directory cleanup: {e}")
    return deleted_count





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

        # Generate a safe, collision-free server-side filename using full UUID hex
        safe_base_name = secure_filename(Path(raw_file_name).stem) or "document"
        unique_file_id = f"{safe_base_name}_{uuid.uuid4().hex}.docx"
        target_docx_path = LOCAL_OUTPUT_DIR / unique_file_id

        # Cleanup expired output files periodically on new conversions
        cleanup_expired_output_files()

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

        # Generate time-limited signed download URL
        expires_at = int(time.time()) + DOWNLOAD_TOKEN_TTL_SECONDS
        download_token = generate_download_token(unique_file_id, expires_at)
        download_url = f"/download/{unique_file_id}?token={download_token}&expires={expires_at}"

        return jsonify({
            "message": "OCR process completed successfully",
            "docx_file": unique_file_id,
            "total_pages": len(page_layouts),
            "download_url": download_url,
            "expires_at": expires_at,
        }), 200

    except SecurityValidationError as sve:
        return jsonify({"error": f"Security validation failed: {str(sve)}"}), 400
    except DocumentProcessingError as dpe:
        logger.error(f"Document processing error: {dpe}")
        return jsonify({"error": "Document processing failed. Please check your document format or quality."}), 500
    except Exception as e:
        logger.exception("Unexpected internal error in /convert")
        return jsonify({"error": "An internal server error occurred while processing the document."}), 500



@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Safely send file with mandatory HMAC signature & expiration verification."""
    safe_name = secure_filename(filename)
    if not safe_name or not safe_name.endswith('.docx'):
        return jsonify({"error": "Invalid file requested."}), 400

    # Verify HMAC token and expiration
    token = request.args.get('token', '')
    raw_expires = request.args.get('expires', '')
    try:
        expires_at = int(raw_expires)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid or missing token expiration."}), 403

    if not verify_download_token(safe_name, expires_at, token):
        return jsonify({"error": "Unauthorized or expired download link."}), 403

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
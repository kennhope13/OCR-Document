import os
import uuid

from google.cloud import storage
from google.auth.credentials import AnonymousCredentials

endpoint = os.environ.get("STORAGE_EMULATOR_HOST")
if not endpoint:
    raise RuntimeError("STORAGE_EMULATOR_HOST is not set")

client = storage.Client(
    project="test-project",
    credentials=AnonymousCredentials(),
)

bucket_name = "ocr-test-bucket"
object_name = f"integration-tests/{uuid.uuid4().hex}.txt"
expected = b"Free local GCS emulator test"

bucket = client.bucket(bucket_name)

try:
    bucket.create()
except Exception as exc:
    from google.api_core.exceptions import Conflict
    if not isinstance(exc, Conflict):
        raise

blob = bucket.blob(object_name)
blob.upload_from_string(expected, content_type="text/plain")
actual = blob.download_as_bytes()

assert actual == expected, "Uploaded and downloaded content differ"

print("PASS: emulator upload/download")
print(f"Bucket: {bucket_name}")
print(f"Object: {object_name}")
